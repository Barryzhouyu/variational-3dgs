import torch
import os
import json
import numpy as np
import torchvision
from PIL import Image
from argparse import ArgumentParser
from os import makedirs

from scene import Scene
from scene.camera_fn import Camera
from gaussian_renderer import render
from scene import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state, PILtoTorch
import torch.nn.functional as F


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def save_perturbed_parameters(gaussians, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(gaussians._xyz.cpu(), os.path.join(output_dir, "xyz.pt"))
    torch.save(gaussians._scaling.cpu(), os.path.join(output_dir, "scaling.pt"))
    torch.save(gaussians._opacity.cpu(), os.path.join(output_dir, "opacity.pt"))
    if hasattr(gaussians, "_features_dc"):
        torch.save(gaussians._features_dc.cpu(), os.path.join(output_dir, "features_dc.pt"))


def bayesian_sample_gaussians(gaussians, xyz_mean, scaling_mean, opacity_mean):
    if hasattr(gaussians, "_xyz_offset") and hasattr(gaussians, "_xyz"):
        std = F.softplus(gaussians._xyz_offset)
        gaussians._xyz = xyz_mean + std * torch.randn_like(xyz_mean)
    if hasattr(gaussians, "_scaling_offset") and hasattr(gaussians, "_scaling"):
        std = F.softplus(gaussians._scaling_offset)
        gaussians._scaling = scaling_mean + std * torch.randn_like(scaling_mean)
    if hasattr(gaussians, "_opacity_offset") and hasattr(gaussians, "_opacity"):
        std = F.softplus(gaussians._opacity_offset)
        gaussians._opacity = opacity_mean + std * torch.randn_like(opacity_mean)


def render_single_view(model_path, iteration, gaussians, pipeline, background, pose_json_path, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, "model_distribution", f"ours_{iteration}")
    makedirs(render_path, exist_ok=True)

    # === Load single frame JSON ===
    with open(pose_json_path, 'r') as f:
        data = json.load(f)

    frame = data["frames"][0]
    intrinsics = data["camera_intrinsics"]["1"]

    # transform_matrix = np.linalg.inv(transform_matrix)
    # R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    # T = torch.tensor(transform_matrix[:3, 3] * -1.0, dtype=torch.float32)

    transform_matrix = np.array(frame["transform_matrix"])
    #transform_matrix = np.linalg.inv(transform_matrix)
    width, height = intrinsics["width"], intrinsics["height"]
    fx, fy = intrinsics["focal_length"]
    FoVx = 2 * np.arctan(width / (2 * fx))
    FoVy = 2 * np.arctan(height / (2 * fy))
    # R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    # T = torch.tensor(transform_matrix[:3, 3], dtype=torch.float32)

    R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    T = torch.tensor(transform_matrix[:3, 3] * -1.0, dtype=torch.float32)


    if torch.cuda.is_available():
        R, T = R.cuda(), T.cuda()

    # Load image and convert to tensor
    image_path = frame["file_path"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((width, height), Image.LANCZOS)
    image_tensor = PILtoTorch(pil_image, (width, height)).cuda()

    # === Camera object ===
    cam = Camera(
        resolution=(width, height),
        colmap_id=frame["camera_id"],
        R=R, T=T,
        FoVx=FoVx, FoVy=FoVy,
        depth_params=None,
        image=image_tensor,
        invdepthmap=None,
        image_name=image_path,
        uid=frame["camera_id"],
        trans=np.array(data.get("scene_center", [0, 0, 0])),
        scale=float(data.get("scene_scale", 1.0)),
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )

    print("Rendering 10 Bayesian samples from single pose...")
    # Save a copy of means to reset before each sample
    xyz_mean = gaussians._xyz.clone()
    scaling_mean = gaussians._scaling.clone()
    opacity_mean = gaussians._opacity.clone()

    for sample_idx in range(10):
        bayesian_sample_gaussians(gaussians, xyz_mean, scaling_mean, opacity_mean)
        rendered = render(cam, gaussians, pipeline, background)["render"]
        out_path = os.path.join(render_path, f"rendered_cus_sample{sample_idx}.png")
        torchvision.utils.save_image(rendered, out_path)
        print(f"Saved rendered image at: {out_path}")
        # ======= Save all parameters for this sample =======
        sample_param_dir = os.path.join(render_path, f"sample_{sample_idx}")
        save_perturbed_parameters(gaussians, sample_param_dir)
        print(f"Saved perturbed parameters at: {sample_param_dir}")

def render_wrapper(dataset: ModelParams, iteration: int, pipeline: PipelineParams, pose_file: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_single_view(
            model_path=dataset.model_path,
            iteration=scene.loaded_iter,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            pose_json_path=pose_file,
            train_test_exp=False,
            separate_sh=SPARSE_ADAM_AVAILABLE
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--pose", type=str, required=True, help="Path to single-pose JSON file")

    args = get_combined_args(parser)
    safe_state(False)

    print("Rendering from:", args.pose)
    render_wrapper(model.extract(args), args.iteration, pipeline.extract(args), args.pose)


    




    


