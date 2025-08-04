import os
import torch
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
import argparse

# === Build parser and add ModelParams ===
parser = argparse.ArgumentParser()
model_params = ModelParams(parser)  # pass parser to ModelParams
opt = parser.parse_args([
    "--model_path", "./output/nb1",
    "--source_path", "/home/roar3/Desktop/nb1/undistorted",
    "--sh_degree", "3"
])

# === Load Gaussian model ===
gaussians = GaussianModel(opt)
sampled = gaussians.sample()

print("✅ Sampled model created.")
print("  Sampled _xyz shape:", sampled._xyz.shape)
print("  Sampled _opacity shape:", sampled._opacity.shape)


