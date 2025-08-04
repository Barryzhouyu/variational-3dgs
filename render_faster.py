import torch
import os
import json
import math
import numpy as np
import torchvision.utils
from PIL import Image
from argparse import ArgumentParser
from os import makedirs

from scene import Scene
from scene.cameras_initial import Camera
from gaussian_renderer import render
from scene import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state, PILtoTorch

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def load_features_dc(gaussians, features_dc_path):
    if os.path.exists(features_dc_path):
        print(f"Loading SH colors from: {features_dc_path}")
        gaussians._features_dc = torch.load(features_dc_path).cuda()
    else:
        raise FileNotFoundError(f"Could not find: {features_dc_path}")


def render_set(model_path, iteration, gaussians, pipeline, background, train_test_exp, separate_sh, custom_pose):
    render_path = os.path.join(model_path, "custom", f"ours_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)

    frame = custom_pose["frames"][0]
    image_path = frame["file_path"]
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist -> {image_path}")
        return

    try:
        pil_image = Image.open(image_path).convert("RGB")
        width = custom_pose["camera_intrinsics"]["1"]["width"]
        height = custom_pose["camera_intrinsics"]["1"]["height"]
        pil_image = pil_image.resize((width, height), Image.LANCZOS)
        image_tensor = PILtoTorch(pil_image, (width, height))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    transform_matrix = np.array(frame["transform_matrix"])
    #transform_matrix = np.linalg.inv(transform_matrix)
    R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    T = torch.tensor(transform_matrix[:3, 3] * -1.0, dtype=torch.float32)
    ##########for rov scene############
    # R = torch.tensor(transform_matrix[:3, :3], dtype=torch.float32)
    # T = torch.tensor(transform_matrix[:3, 3], dtype=torch.float32)

    if torch.cuda.is_available():
        R, T, image_tensor = R.cuda(), T.cuda(), image_tensor.cuda()

    fx, fy = custom_pose["camera_intrinsics"]["1"]["focal_length"]
    FoVx = 2 * math.atan(width / (2 * fx))
    FoVy = 2 * math.atan(height / (2 * fy))

    custom_camera = Camera(
        resolution=(width, height),
        colmap_id=frame["camera_id"],
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=image_tensor,
        invdepthmap=None,
        image_name=frame["file_path"],
        uid=frame["camera_id"],
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )

    print("Rendering...")
    rendering = render(custom_camera, gaussians, pipeline, background)["render"]
    output_path = os.path.join(render_path, "custom_render.png")
    torchvision.utils.save_image(rendering, output_path)
    print(f"Rendered image saved at: {output_path}")
    
    
def render_sets_fast(
    model_path, iteration, gaussians, pipeline, background,
    custom_pose_file, features_dc_path, xyz_path=None,
    scaling_path=None, opacity_path=None,
    separate_sh=False, train_test_exp=False
):
    with torch.no_grad():
        with open(custom_pose_file, "r") as f:
            custom_pose = json.load(f)

        if "frames" not in custom_pose or "camera_intrinsics" not in custom_pose:
            raise ValueError("Custom pose file missing required keys!")

        # Load features_dc once per perturb
        if features_dc_path is not None:
            gaussians._features_dc = torch.load(features_dc_path).cuda()

        if xyz_path is not None:
            gaussians._xyz = torch.load(xyz_path).cuda()

        if scaling_path is not None:
            gaussians._scaling = torch.load(scaling_path).cuda()

        if opacity_path is not None:
            gaussians._opacity = torch.load(opacity_path).cuda()


        render_set(model_path, iteration, gaussians, pipeline, background,
                   train_test_exp, separate_sh, custom_pose)



def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, separate_sh: bool, custom_pose_file=None, features_dc_path=None, xyz_path=None):
    with torch.no_grad():
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        if custom_pose_file is None or features_dc_path is None:
            raise ValueError("Missing required --pose or --features_dc arguments!")

        with open(custom_pose_file, "r") as f:
            custom_pose = json.load(f)

        if "frames" not in custom_pose or "camera_intrinsics" not in custom_pose:
            raise ValueError("Custom pose file missing required keys!")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Load features_dc
        load_features_dc(gaussians, features_dc_path)

        if xyz_path is not None:
            #print(f"Loading perturbed positions from: {xyz_path}")
            gaussians._xyz = torch.load(xyz_path).cuda()

        render_set(dataset.model_path, scene.loaded_iter, gaussians, pipeline, background, dataset.train_test_exp, separate_sh, custom_pose)



if __name__ == "__main__":
    parser = ArgumentParser(description="Render custom pose using perturbed SH colors")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load the model from")
    parser.add_argument("--pose", type=str, required=True, help="Path to JSON file containing custom camera pose")
    parser.add_argument("--features_dc", type=str, required=True, help="Path to features_dc.pt file to load")
    parser.add_argument("--xyz", type=str, required=False, help="Path to xyz.pt (position noise)")
    parser.add_argument('--scaling', type=str, default=None, help='Path to scaling.pt file')
    parser.add_argument('--opacity', type=str, default=None, help='Path to opacity.pt file')


    args = get_combined_args(parser)
    print("Rendering from model path:", args.model_path)

    safe_state(False)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        separate_sh=SPARSE_ADAM_AVAILABLE,
        custom_pose_file=args.pose,
        features_dc_path=args.features_dc,
        xyz_path=args.xyz,
        scaling_path=args.scaling,        
        opacity_path=args.opacity 
    )

