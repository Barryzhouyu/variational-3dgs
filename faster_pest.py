import json
import torch
import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene
from render_faster import render_sets_fast

# ========== Constants ==========
# MODEL_PATH = "/home/roar3/variational-3dgs/output/hope_2/"
# POSE_JSON_PATH = "/home/roar3/Desktop/guessed_pose.json"
# POINT_CLOUD_PATH = "/home/roar3/variational-3dgs/output/hope_2/point_cloud/iteration_10000/point_cloud.ply"
# RENDERED_IMAGE_PATH = "/home/roar3/variational-3dgs/output/hope_2/custom/ours_10000/renders/custom_render.png"
# IMAGE_DIR = "/home/roar3/Desktop/final"
# PLOT_BASE_DIR = "/home/roar3/Desktop/final_plot"
# ITERATION = 10000

MODEL_PATH = "/home/roar3/variational-3dgs/output/pool/"
POSE_JSON_PATH = "/home/roar3/Desktop/guessed_pose.json"
POINT_CLOUD_PATH = "/home/roar3/variational-3dgs/output/pool/point_cloud/iteration_20000/point_cloud.ply"
RENDERED_IMAGE_PATH = "/home/roar3/variational-3dgs/output/pool/custom/ours_20000/renders/custom_render.png"
IMAGE_DIR = "/home/roar3/Desktop/pool_plot"
PLOT_BASE_DIR = "/home/roar3/Desktop/pool_plot"
ITERATION = 20000


# camera_matrix = np.array([
#     [3576.8065146501135, 0, 960],
#     [0, 3576.8065146501135, 640],
#     [0, 0, 1]
# ], dtype=np.float32)

camera_matrix = np.array([
    [1068.920574552045, 0, 960],
    [0, 1068.920574552045, 540],
    [0, 0, 1]
], dtype=np.float32)

R_gt = np.array([
        [0.9995760301076153, -0.0019346787482901427, -0.029051971575793496],
        [0.002143888367292467, 0.9999719841343944,    0.00717179816998596],
        [0.02903728253415575, -0.007231041727395881,  0.9995521738551563]
    ])

t_gt = np.array([
        3.729789058561073,
        1.2698616033619443,
    -0.4110934986660983
    ])

# === Setup parser (following render_cus_2 style) ===
parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)

# === Required Arguments ===
parser.add_argument('--depths', type=str, default=None, help='Path to depths file')
parser.add_argument("--iteration", type=int, default=ITERATION)
parser.add_argument("--pose", type=str, default=POSE_JSON_PATH)
parser.add_argument("--features_dc", type=str)
parser.add_argument("--xyz", type=str)

# === Simulated CLI Input ===
args = parser.parse_args([
    "--model_path", MODEL_PATH,
    "--source_path", "/home/roar3/Desktop/pool/undistorted",
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
gaussians = GaussianModel(model_args)
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
    orb = cv2.ORB_create(nfeatures=5000)  # Increased features
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


# def plot_pose_error_trajectory(rvec_list, tvec_list, gt_json_path, frame_id, perturb_id, plot_base_dir):
#     """
#     Plots Euler angle errors (roll, pitch, yaw) and translation L2 error over iterations,
#     using the ground-truth pose from the transforms.json file. Also prints matched image.
#     """
#     # === Load GT pose from JSON ===
#     with open(gt_json_path, "r") as f:
#         gt_data = json.load(f)

#     # Expected filenames to match
#     expected_filenames = [f"{frame_id:03d}.png", f"frame_{frame_id:03d}.png"]

#     R_gt = None
#     t_gt = None

#     for frame in gt_data["frames"]:
#         fname = os.path.basename(frame["file_path"])
#         if any(fname.endswith(name) for name in expected_filenames):
#             matrix = np.array(frame["transform_matrix"])
#             R_gt = matrix[:3, :3]
#             t_gt = matrix[:3, 3]
#             print(f"Matched GT pose with image name: {fname}")
#             break

#     if R_gt is None or t_gt is None:
#         raise ValueError(f"GT pose for frame {frame_id} not found in {gt_json_path}")

#     # === Convert GT rotation to Euler ===
#     euler_gt = R.from_matrix(R_gt).as_euler('xyz', degrees=True)

#     # === Containers for error over time ===
#     roll_errors = []
#     pitch_errors = []
#     yaw_errors = []
#     translation_errors = []

#     for rvec, tvec in zip(rvec_list, tvec_list):
#         R_est, _ = cv2.Rodrigues(rvec)
#         euler_est = R.from_matrix(R_est).as_euler('xyz', degrees=True)
#         angle_error = euler_est - euler_gt

#         # Absolute angle error
#         roll_errors.append(abs(angle_error[0]))
#         pitch_errors.append(abs(angle_error[1]))
#         yaw_errors.append(abs(angle_error[2]))

#         # Translation error
#         t_est = np.array(tvec).flatten()
#         trans_error = np.linalg.norm(t_est - t_gt)
#         translation_errors.append(trans_error)

#     # === Plotting ===
#     iterations = range(1, len(rvec_list) + 1)
#     plt.figure(figsize=(8, 6))

#     # Rotation errors
#     plt.subplot(2, 1, 1)
#     plt.plot(iterations, roll_errors, label='Roll Error (°)', marker='o', color='orange', linewidth=3, markersize=8)
#     plt.plot(iterations, pitch_errors, label='Pitch Error (°)', marker='o', color='green', linewidth=3, markersize=8)
#     plt.plot(iterations, yaw_errors, label='Yaw Error (°)', marker='o', color='blue', linewidth=3, markersize=8)
#     plt.ylabel("Euler Angle Error (degrees)")
#     plt.title("Rotation Errors")
#     plt.legend()
#     plt.grid(True)

#     # Translation errors
#     plt.subplot(2, 1, 2)
#     plt.plot(iterations, translation_errors, label='Translation Error (L2)', marker='o', color='hotpink', linewidth=3, markersize=8)
#     plt.xlabel("Iteration")
#     plt.ylabel("Translation Error")
#     plt.title("Translation Error over Iterations")
#     plt.grid(True)

#     plt.tight_layout()

#     # === Save Plot ===
#     save_dir = os.path.join(plot_base_dir, f"norm_plot_{frame_id:03d}")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"noisy_pose_error_custom_sample_{perturb_id}.png")
#     plt.savefig(save_path)

def plot_pose_error_trajectory(rvec_list, tvec_list):
    """
    Plots Euler angle errors (roll, pitch, yaw) and translation L2 error over iterations.
    """
    
    euler_gt = R.from_matrix(R_gt).as_euler('xyz', degrees=True)

    # === Containers for error over time ===
    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    translation_errors = []

    for rvec, tvec in zip(rvec_list, tvec_list):
        # Rotation matrix from rvec
        R_est, _ = cv2.Rodrigues(rvec)
        euler_est = R.from_matrix(R_est).as_euler('xyz', degrees=True)
        angle_error = euler_est - euler_gt

        # Absolute angle error per axis
        roll_errors.append(abs(angle_error[0]))
        pitch_errors.append(abs(angle_error[1]))
        yaw_errors.append(abs(angle_error[2]))

        # Translation error
        t_est = np.array(tvec).flatten()
        trans_error = np.linalg.norm(t_est - t_gt)
        translation_errors.append(trans_error)

    # === Plotting ===
    iterations = range(1, len(rvec_list) + 1)

    plt.figure(figsize=(8, 6))

    # Rotation error subplot
    plt.subplot(2, 1, 1)
    plt.plot(iterations, roll_errors, label='Roll Error (°)', marker='o', color='orange', linewidth=3, markersize=8)
    plt.plot(iterations, pitch_errors, label='Pitch Error (°)', marker='o', color='green', linewidth=3, markersize=8)
    plt.plot(iterations, yaw_errors, label='Yaw Error (°)', marker='o', color='blue', linewidth=3, markersize=8)
    plt.ylabel("Euler Angle Error (degrees)")
    plt.title("Rotation Errors")
    plt.legend()
    plt.grid(True)

    # Translation error subplot
    plt.subplot(2, 1, 2)
    plt.plot(iterations, translation_errors, label='Translation Error (L2)', marker='o', color='hotpink', linewidth=3, markersize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Translation Error")
    plt.title("Translation Error over Iterations")
    plt.grid(True)

    plt.tight_layout()

    # === Save figure ===
    #save_path = f"/home/roar/Desktop/coke_plot/norm_plot_068/pose_error_custom_sample_{perturb_id}.png"
    frame_id = int(Path(ACTUAL_IMAGE_PATH).stem.split("_")[1])
    #frame_id = int(Path(ACTUAL_IMAGE_PATH).stem)
    save_dir = os.path.join(PLOT_BASE_DIR, f"norm_plot_{frame_id:03d}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_{perturb_id}.png")
    plt.savefig(save_path)
    print(f"Saved pose error plot to {save_path}")

    #plt.show()
    
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

    features_dc_path = os.path.join(MODEL_PATH, f"sample_{perturb_id}", "features_dc.pt")
    xyz_path = os.path.join(MODEL_PATH, f"sample_{perturb_id}", "xyz.pt")
    scaling_path = os.path.join(MODEL_PATH, f"sample_{perturb_id}", "scaling.pt") 
    opacity_path = os.path.join(MODEL_PATH, f"sample_{perturb_id}", "opacity.pt")


    for i in range(iterations):
        print(f"\n Iteration {i+1}/{iterations}: Rendering with sample_{perturb_id}")

        render_sets_fast(
            model_path=MODEL_PATH,
            iteration=ITERATION,
            gaussians=gaussians,
            pipeline=pipeline_args,
            background=background,
            custom_pose_file=POSE_JSON_PATH,
            features_dc_path=features_dc_path,
            xyz_path=xyz_path,
            scaling_path=scaling_path,
            opacity_path=opacity_path,
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

            # if prev_loss is not None and i >= MIN_ITERATIONS:
            #     loss_delta = abs(prev_loss - loss)
            #     if loss_delta < LOSS_STOP_THRESHOLD:
            #         no_improvement_count += 1
            #         print(f"Stable loss for {no_improvement_count} iterations")
            #     else:
            #         no_improvement_count = 0

            #     if no_improvement_count >= 3:
            #         print(" Early stopping: Loss stabilized")
            #         break
            
            t_est = np.array(tvec).flatten()
            trans_error = np.linalg.norm(t_est - t_gt)
            translation_errors.append(trans_error)

            if prev_trans_error is not None and i >= MIN_ITERATIONS:
                delta_error = abs(prev_trans_error - trans_error)
                print(delta_error)
                if delta_error < LOSS_STOP_THRESHOLD:
                    no_improvement_count += 1
                    print(f"Stable translation error for {no_improvement_count} iterations")
                else:
                    no_improvement_count = 0

                if no_improvement_count >= 3:
                    print("Early stopping: Translation error stabilized")
                    break

            prev_trans_error = trans_error

            R_mat, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = R_mat
            transform[:3, 3] = tvec.flatten()
            pose_data["frames"][0]["transform_matrix"] = transform.tolist()

            with open(POSE_JSON_PATH, "w") as f:
                json.dump(pose_data, f, indent=4)

            frame_id = int(Path(ACTUAL_IMAGE_PATH).stem.split("_")[1])
            pose_output_dir = f"/home/roar3/variational-3dgs/output/pool/est_poses/f_{frame_id:03d}_Epose"
            os.makedirs(pose_output_dir, exist_ok=True)
            output_pose_path = os.path.join(pose_output_dir, f"pose_custom_sample_{perturb_id}.json")
            with open(output_pose_path, "w") as f:
                json.dump(pose_data, f, indent=4)
            print(f"Saved estimated pose: {output_pose_path}")
        else:
            print("PnP Failed")

    plot_loss(loss_history)
    plot_pose_error_trajectory(rvec_list, tvec_list)
#     plot_pose_error_trajectory(
#     rvec_list,
#     tvec_list,
#     gt_json_path="/home/roar/Desktop/h_n/transforms.json",
#     frame_id=frame_id,
#     perturb_id=perturb_id,      
#     plot_base_dir=PLOT_BASE_DIR    
# )
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

for fname in image_files:
    frame_num = int(fname.split("_")[1].split(".")[0])
    ACTUAL_IMAGE_PATH = os.path.join(IMAGE_DIR, fname)
    globals()["ACTUAL_IMAGE_PATH"] = ACTUAL_IMAGE_PATH

    print(f"\n==============================")
    print(f"Processing frame: {fname}")
    print(f"==============================")

    for perturb_id in range(0, 10):
        print(f"\n→ Starting pose optimization for custom_sample_{perturb_id} (frame_{frame_num:03d})")

        os.system("python /home/roar3/variational-3dgs/guessed_pose_json_create.py")

        try:
            optimize_pose_pnp(perturb_id=perturb_id, iterations=50)
        except cv2.error:
            print(f"OpenCV error on sample {perturb_id} (frame {frame_num}), skipping...\n")
            continue
        except Exception as e:
            print(f"Unexpected error on sample {perturb_id} (frame {frame_num}): {e}")
            print("Skipping and continuing...\n")
            continue

