import json
import torch
import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene
from render_cus_3 import render_sets_fast

# ========== Constants ==========
MODEL_PATH = "/home/roar/gaussian-splatting/output/save/"
POSE_JSON_PATH = "/home/roar/Desktop/guessed_pose.json"
POINT_CLOUD_PATH = "/home/roar/gaussian-splatting/output/save/point_cloud/iteration_20000/point_cloud.ply"
RENDERED_IMAGE_PATH = "/home/roar/gaussian-splatting/output/save/custom/ours_20000/renders/custom_render.png"
IMAGE_DIR = "/home/roar/Desktop/for_plot"
PLOT_BASE_DIR = "/home/roar/Desktop/rov_plot_better"
ITERATION = 20000

camera_matrix = np.array([
    [434.01818810151997, 0, 959.5],
    [0, 434.01818810151997, 639.5],
    [0, 0, 1]
], dtype=np.float32)

# R_gt = np.array([
#         [0.9995760301076153, -0.0019346787482901427, -0.029051971575793496],
#         [0.002143888367292467, 0.9999719841343944,    0.00717179816998596],
#         [0.02903728253415575, -0.007231041727395881,  0.9995521738551563]
#     ])

# t_gt = np.array([
#         3.729789058561073,
#         1.2698616033619443,
#     -0.4110934986660983
#     ])

# === Setup parser (following render_cus_2 style) ===
parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

# === Required Arguments ===
parser.add_argument("--iteration", type=int, default=ITERATION)
parser.add_argument("--pose", type=str, default=POSE_JSON_PATH)
parser.add_argument("--features_dc", type=str)
parser.add_argument("--xyz", type=str)

# === Simulated CLI Input ===
args = parser.parse_args([
    "--model_path", MODEL_PATH,
    "--source_path", "/home/roar/Desktop/nb/undistorted",
    "--images", "images",
    "--depths", "",
    "--resolution", "1",
    "--data_device", "cuda",
    "--sh_degree", "3"
])


# === Extract config ===
model_args = model.extract(args)
pipeline_args = pipeline.extract(args)

# === Load Gaussians and Scene ===
gaussians = GaussianModel(model_args.sh_degree)
scene = Scene(model_args, gaussians, load_iteration=ITERATION, shuffle=False)
background = torch.tensor([1, 1, 1] if model_args.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")

def load_pose(json_path):
    """Load rvec & tvec from guessed pose JSON file."""
    with open(json_path, "r") as f:
        pose_data = json.load(f)

    transform_matrix = np.array(pose_data["frames"][0]["transform_matrix"])
    R = transform_matrix[:3, :3]
    tvec = transform_matrix[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to vector

    return rvec, tvec, pose_data

# 🔹 Load 3D Point Cloud
def load_point_cloud(ply_path):
    """Loads a point cloud from a PLY file and returns an Nx3 NumPy array."""
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points)  # Extracts the 3D points as a NumPy array

# 🔹 Project 3D Point Cloud to 2D
def project_3d_to_2d(pts_3d, camera_matrix, rvec, tvec):
    """Projects 3D points into 2D image space using camera parameters."""
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
    return pts_2d.squeeze()  # Remove extra dimensions

# 🔹 Extract 2D-2D Correspondences Using Feature Matching
def match_features():
    """Match features between actual and rendered images using ORB."""
    actual_img = cv2.imread(ACTUAL_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    rendered_img = cv2.imread(RENDERED_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=2000)  # Increased features
    kp1, des1 = orb.detectAndCompute(actual_img, None)
    kp2, des2 = orb.detectAndCompute(rendered_img, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)[:500]  # More matches

    # Extract corresponding 2D points
    pts_actual = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_rendered = np.float32([kp2[m.trainIdx].pt for m in matches])

    print(f"Matched {len(pts_actual)} feature points")

    return pts_actual, pts_rendered

# 🔹 Convert 2D-2D Matches to 2D-3D Correspondences
def get_2d_3d_correspondences(pts_actual, pts_rendered, camera_matrix, rvec, tvec, point_cloud):
    """Converts 2D-2D correspondences into 2D-3D pairs using the point cloud."""
    projected_2d = project_3d_to_2d(point_cloud, camera_matrix, rvec, tvec)

    pts_3d = []
    pts_2d = []
    seen_3d_points = set()  # Prevent duplicate 3D matches

    for i in range(len(pts_rendered)):
        distances = np.linalg.norm(projected_2d - pts_rendered[i], axis=1)
        nearest_idx = np.argmin(distances)
        nearest_3d = tuple(point_cloud[nearest_idx])  # Convert to tuple for set operations

        if nearest_3d not in seen_3d_points:
            seen_3d_points.add(nearest_3d)
            pts_3d.append(point_cloud[nearest_idx])
            pts_2d.append(pts_actual[i])

    return np.array(pts_2d, dtype=np.float32), np.array(pts_3d, dtype=np.float32)

def compute_reprojection_error(pts_2d, pts_3d, camera_matrix, rvec, tvec):
    """
    Computes the reprojection error by projecting 3D points into 2D and
    measuring the difference with actual 2D matches.
    """
    projected_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
    projected_pts = projected_pts.squeeze()

    # Compute Euclidean distance between projected and actual 2D points
    error = np.linalg.norm(pts_2d - projected_pts, axis=1)
    mean_error = np.mean(error)

    return mean_error

def plot_pose_error_trajectory(rvec_list, tvec_list, R_gt, t_gt, frame_id, perturb_id, plot_base_dir):
    # Convert GT rotation matrix to Euler angles
    euler_gt = R.from_matrix(R_gt).as_euler('xyz', degrees=True)

    # Error containers
    roll_errors, pitch_errors, yaw_errors = [], [], []
    translation_errors = []

    for rvec, tvec in zip(rvec_list, tvec_list):
        R_est, _ = cv2.Rodrigues(rvec)
        euler_est = R.from_matrix(R_est).as_euler('xyz', degrees=True)
        euler_error = euler_est - euler_gt

        roll_errors.append(abs(euler_error[0]))
        pitch_errors.append(abs(euler_error[1]))
        yaw_errors.append(abs(euler_error[2]))

        t_est = np.array(tvec).flatten()
        translation_errors.append(np.linalg.norm(t_est - t_gt))

    # Plot
    iterations = range(1, len(rvec_list) + 1)
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(iterations, roll_errors, label='Roll Error (°)', marker='o', color='orange', linewidth=3, markersize=6)
    plt.plot(iterations, pitch_errors, label='Pitch Error (°)', marker='o', color='green', linewidth=3, markersize=6)
    plt.plot(iterations, yaw_errors, label='Yaw Error (°)', marker='o', color='blue', linewidth=3, markersize=6)
    plt.ylabel("Euler Angle Error (degrees)")
    plt.title("Rotation Errors")
    plt.legend()
    plt.grid(True)

    # Translation error subplot
    plt.subplot(2, 1, 2)
    plt.plot(iterations, translation_errors, label='Translation Error (L2)', marker='o', color='hotpink', linewidth=3, markersize=6)
    plt.xlabel("Iteration")
    plt.ylabel("Translation Error")
    plt.title("Translation Error over Iterations")
    plt.grid(True)

    plt.tight_layout()

    # Save
    save_dir = os.path.join(plot_base_dir, f"norm_plot_{frame_id:03d}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"noisy_pose_error_custom_sample_{perturb_id}.png")
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    
    
def optimize_pose_pnp(perturb_id, iterations=50):
    rvec, tvec, pose_data = load_pose(POSE_JSON_PATH)
    point_cloud = load_point_cloud(POINT_CLOUD_PATH)

    loss_history = []
    rvec_list = []
    tvec_list = []
    prev_trans_error = None
    no_improvement_count = 0
    LOSS_STOP_THRESHOLD = 1e-2
    MIN_ITERATIONS = 5
    translation_errors = []

    features_dc_path = os.path.join(MODEL_PATH, f"custom_sample_{perturb_id}", "features_dc.pt")
    xyz_path = os.path.join(MODEL_PATH, f"custom_sample_{perturb_id}", "xyz.pt")

    for i in range(iterations):
        print(f"\n Iteration {i+1}/{iterations}: Rendering with custom_sample_{perturb_id}")

        render_sets_fast(
            model_path=MODEL_PATH,
            iteration=ITERATION,
            gaussians=gaussians,
            pipeline=pipeline_args,
            background=background,
            custom_pose_file=POSE_JSON_PATH,
            features_dc_path=features_dc_path,
            xyz_path=xyz_path,
            separate_sh=False,
            train_test_exp=False
        )

        # The rest of the optimization loop remains unchanged
        pts_actual, pts_rendered = match_features()
        pts_2d, pts_3d = get_2d_3d_correspondences(
            pts_actual, pts_rendered, camera_matrix, rvec, tvec, point_cloud
        )

        if len(pts_2d) < 4:
            print("Not enough correspondences. Skipping iteration.")
            continue

        success, rvec_new, tvec_new, _ = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera_matrix, None, reprojectionError=15.0, iterationsCount=10000
        )

        if not success:
            print("RANSAC Failed. Trying normal PnP...")
            success, rvec_new, tvec_new = cv2.solvePnP(pts_3d, pts_2d, camera_matrix, None)

        if success:
            alpha = 0.1
            rvec = (1 - alpha) * rvec + alpha * rvec_new
            tvec = (1 - alpha) * tvec + alpha * tvec_new

            rvec_list.append(rvec.flatten())
            tvec_list.append(tvec.flatten())

            loss = compute_reprojection_error(pts_2d, pts_3d, camera_matrix, rvec, tvec)
            loss_history.append(loss)
            print(f" Reprojection Loss: {loss:.4f} pixels")
            
            t_est = np.array(tvec).flatten()
            trans_error = np.linalg.norm(t_est - t_gt)
            translation_errors.append(trans_error)

            if prev_trans_error is not None and i >= MIN_ITERATIONS:
                delta_error = prev_trans_error - trans_error
                print(f"[{i+1}] trans_error: {trans_error:.4f}, delta: {delta_error:.4f}")
                if delta_error > 0 and abs(delta_error) < LOSS_STOP_THRESHOLD:
                    no_improvement_count += 1
                    print(f"Stable and improving for {no_improvement_count} iterations")
                else:
                    no_improvement_count = 0
                    print("Reset early stop counter due to increase or large drop")

                if no_improvement_count >= 3:
                    print("Early stopping: Translation error stabilized")
                    break

            prev_trans_error = trans_error

            R_mat, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = R_mat
            transform[:3, 3] = tvec.flatten()
            transform_inv = np.linalg.inv(transform)

            pose_data["frames"][0]["transform_matrix"] = transform_inv.tolist()

            with open(POSE_JSON_PATH, "w") as f:
                json.dump(pose_data, f, indent=4)

            frame_id = int(Path(ACTUAL_IMAGE_PATH).stem.split("_")[1])
            pose_output_dir = f"/home/roar/gaussian-splatting/output/save/plot_est/f_{frame_id:03d}_Epose"
            os.makedirs(pose_output_dir, exist_ok=True)
            output_pose_path = os.path.join(pose_output_dir, f"pose_custom_sample_{perturb_id}.json")
            with open(output_pose_path, "w") as f:
                json.dump(pose_data, f, indent=4)
            print(f"Saved estimated pose: {output_pose_path}")
        else:
            print("PnP Failed")

    plot_loss(loss_history)
 
    # Then call the function:
    plot_pose_error_trajectory(
        rvec_list,
        tvec_list,
        R_gt,
        t_gt,
        frame_id=frame_id,
        perturb_id=perturb_id,
        plot_base_dir=PLOT_BASE_DIR
    )
    print(f"Optimization complete for custom_sample_{perturb_id}")


def plot_loss(loss_history):
    """Plots the reprojection loss vs. iteration."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection Loss (pixels)")
    plt.title("Reprojection Loss vs Iteration")
    plt.grid(True)
    #plt.show()


image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.startswith("frame_") and f.endswith(".png")])

# === Load coordinate-to-frame_id mapping once ===
coord_to_id = {}
with open("/home/roar/Desktop/test_nb/trans", "r") as f:
    for line in f:
        if "Renamed:" in line:
            old, new = line.strip().split("Renamed: ")[1].split(" -> ")
            frame_id = int(new.split("_")[1].split(".")[0])
            coord_to_id[old] = frame_id

id_to_coord = {v: k for k, v in coord_to_id.items()} 

for fname in image_files:
    frame_num = int(fname.split("_")[1].split(".")[0])
    ACTUAL_IMAGE_PATH = os.path.join(IMAGE_DIR, fname)
    globals()["ACTUAL_IMAGE_PATH"] = ACTUAL_IMAGE_PATH
    

    # === Load GT pose for current frame ===
    coord_name = id_to_coord.get(frame_num)
    if coord_name is None:
        print(f"Skipping frame {frame_num}: no matching GT coord found")
        continue

    with open("/home/roar/gaussian-splatting/output/save/cameras.json", "r") as f:
        gt_data = json.load(f)

    R_gt, t_gt = None, None
    for frame in gt_data:
        if os.path.basename(frame["img_name"]) == coord_name:
            matrix = np.array(frame["transform_matrix"])
            R_gt = matrix[:3, :3]
            t_gt = matrix[:3, 3]
            break

    if t_gt is None:
        print(f"Skipping frame {frame_num}: GT pose not found")
        continue
    
    print(f"\n==============================")
    print(f"Processing frame: {fname}")
    print(f"==============================")

    for perturb_id in range(1, 11):
        print(f"\n→ Starting pose optimization for custom_sample_{perturb_id} (frame_{frame_num:03d})")

        os.system("python /home/roar/Desktop/3DGS_Dataset_creation/guessed_pose_json_create.py")

        try:
            optimize_pose_pnp(perturb_id=perturb_id, iterations=50)
        except cv2.error:
            print(f"OpenCV error on sample {perturb_id} (frame {frame_num}), skipping...\n")
            continue
        except Exception as e:
            print(f"Unexpected error on sample {perturb_id} (frame {frame_num}): {e}")
            print("Skipping and continuing...\n")
            continue

