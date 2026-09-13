import os
import random
import shutil

# Source folder containing all images
source_dir = "dataset/manual_data/Non"

# Destination folders
train_dir = "dataset/manual_data/train1/REAL"
val_dir = "dataset/manual_data/val1/REAL"

# Create folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Get image files
image_files = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith(image_extensions)
]

# Shuffle files
random.shuffle(image_files)

# Split 80-20
split_idx = int(len(image_files) * 0.9)

train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Move files
for file in train_files:
    shutil.copy2(
        os.path.join(source_dir, file),
        os.path.join(train_dir, file)
    )

for file in val_files:
    shutil.copy2(
        os.path.join(source_dir, file),
        os.path.join(val_dir, file)
    )

print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")