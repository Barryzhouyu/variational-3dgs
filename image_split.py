import os
import json
from glob import glob

# --- Config ---
IMAGE_DIR = "/home/roar3/Desktop/nb2/undistorted/images"
IMAGE_EXT = ".png"
TRAIN_RATIO = 0.8

# Step 1: list and sort images (by filename)
image_files = sorted(glob(os.path.join(IMAGE_DIR, "*" + IMAGE_EXT)))

print(f"Found {len(image_files)} images")

# Step 2: generate train/test indices
n_images = len(image_files)
train_count = int(n_images * TRAIN_RATIO)

train_files = image_files[:train_count]
test_files = image_files[train_count:]

print(f"Train: {len(train_files)} images")
print(f"Test: {len(test_files)} images")

# Step 3: save split json with filenames (not indices)
split_data = {
    "train_files": [os.path.basename(f) for f in train_files],
    "test_files": [os.path.basename(f) for f in test_files]
}

split_file = os.path.join(IMAGE_DIR, "train_test_split.json")
with open(split_file, "w") as f:
    json.dump(split_data, f, indent=4)

print(f"Saved split file to {split_file}")

