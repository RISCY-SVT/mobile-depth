import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Path settings
data_root = "..\MidAir_dataset"
dataset = "Kite_training"
environment = "sunny"
depth_scale = 100.0  # Same as in your training

# Find some image pairs
color_dir = os.path.join(data_root, dataset, environment, 'color_left')
depth_dir = os.path.join(data_root, dataset, environment, 'depth')

# Get the first trajectory
trajectory = next(os.walk(color_dir))[1][0]  # First subdirectory
rgb_path = os.path.join(color_dir, trajectory)
depth_path = os.path.join(depth_dir, trajectory)

# Create debug directory
os.makedirs('debug_vis', exist_ok=True)

# Find image files
rgb_files = []
for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
    import glob
    rgb_files.extend(glob.glob(os.path.join(rgb_path, ext)))

# Process first 5 images
for i, rgb_file in enumerate(rgb_files[:5]):
    # Get file name and corresponding depth file
    file_id = os.path.splitext(os.path.basename(rgb_file))[0]
    depth_file = os.path.join(depth_path, f"{file_id}.png")
    
    if not os.path.exists(depth_file):
        print(f"Depth file not found: {depth_file}")
        continue
    
    # Load images
    rgb_image = Image.open(rgb_file).convert('RGB')
    depth_image = Image.open(depth_file)
    
    # Convert to numpy for analysis
    depth_raw = np.array(depth_image)
    rgb_raw = np.array(rgb_image)
    
    print(f"\nImage pair {i+1}:")
    print(f"RGB: {rgb_file}")
    print(f"Depth: {depth_file}")
    print(f"RGB shape: {rgb_raw.shape}, Depth shape: {depth_raw.shape}")
    print(f"Depth dtype: {depth_raw.dtype}, min: {depth_raw.min()}, max: {depth_raw.max()}, mean: {depth_raw.mean():.2f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # RGB image
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_raw)
    plt.title("RGB Image")
    plt.axis('off')
    
    # Raw depth map
    plt.subplot(2, 2, 2)
    plt.imshow(depth_raw)
    plt.colorbar()
    plt.title(f"Raw Depth (min={depth_raw.min()}, max={depth_raw.max()})")
    plt.axis('off')
    
    # Normalized depth (similar to your normalization)
    depth_norm = depth_raw.astype(float) / depth_scale
    depth_norm = np.clip(depth_norm, 0, 1)
    
    plt.subplot(2, 2, 3)
    plt.imshow(depth_norm, cmap='plasma')
    plt.colorbar()
    plt.title(f"Normalized Depth (scale={depth_scale})")
    plt.axis('off')
    
    # Histogram of depth values
    plt.subplot(2, 2, 4)
    plt.hist(depth_raw.flatten(), bins=50)
    plt.title("Depth Histogram")
    
    plt.tight_layout()
    plt.savefig(f"debug_vis/depth_analysis_{i+1}.png")
    plt.close('all')  # Explicitly close all figures to avoid memory leaks

print("Debug visualizations saved to 'debug_vis' directory")
