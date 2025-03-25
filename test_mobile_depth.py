import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path
from collections import defaultdict
import glob

# Import the model architecture
from depth_model import MobileDepth


class TestDataset(Dataset):
    """Dataset for testing the depth model on MidAir test data"""
    def __init__(self, root_dir, dataset_type='Kite_test', environments=None, 
                image_size=384, max_samples=None):
        """
        Initialize the test dataset
        
        Args:
            root_dir: Root directory of the MidAir dataset
            dataset_type: Dataset type ('Kite_test', 'PLE_test', or 'VO_test')
            environments: List of environments to include (e.g., ['sunny', 'foggy'])
            image_size: Size to resize images to
            max_samples: Maximum number of samples to use (for quick testing)
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.image_size = image_size
        
        if environments is None:
            if dataset_type == 'Kite_test':
                self.environments = ['sunny', 'foggy', 'cloudy', 'sunset']
            elif dataset_type == 'PLE_test':
                self.environments = ['fall', 'spring', 'winter']
            elif dataset_type == 'VO_test':
                self.environments = ['sunny', 'foggy', 'sunset']
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
        else:
            self.environments = environments
        
        # Set up image paths
        self.image_paths = []
        self.env_trajectory_map = defaultdict(list)
        
        # Find all test images
        print(f"Searching for test images in {dataset_type}...")
        for env in self.environments:
            env_path = os.path.join(self.root_dir, dataset_type, env, 'color_left')
            
            if not os.path.exists(env_path):
                print(f"Warning: Environment path not found: {env_path}")
                continue
                
            trajectory_dirs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
            
            for traj in trajectory_dirs:
                traj_path = os.path.join(env_path, traj)
                
                # Find all images in the trajectory
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
                    image_files.extend(glob.glob(os.path.join(traj_path, ext)))
                
                image_files = sorted(image_files)
                
                # Track images by environment and trajectory
                self.env_trajectory_map[(env, traj)].extend(image_files)
                self.image_paths.extend(image_files)
        
        # Limit sample size if specified
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]
            
            # Update env_trajectory_map to match limited samples
            new_env_traj_map = defaultdict(list)
            for img_path in self.image_paths:
                env = os.path.normpath(img_path).split(os.sep)[-4]  # Extract env from path
                traj = os.path.normpath(img_path).split(os.sep)[-2]  # Extract trajectory from path
                new_env_traj_map[(env, traj)].append(img_path)
                
            self.env_trajectory_map = new_env_traj_map
        
        # Set up transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Found {len(self.image_paths)} test images across {len(self.env_trajectory_map)} trajectories")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Extract environment and trajectory information
        parts = os.path.normpath(img_path).split(os.sep)
        env = parts[-4]  # Environment is 4 levels up from the image file
        trajectory = parts[-2]  # Trajectory is 2 levels up
        filename = parts[-1]  # Just the filename
        
        return {
            'image': image_tensor,
            'path': img_path,
            'environment': env,
            'trajectory': trajectory,
            'filename': filename
        }
    
    def get_trajectory_samples(self, environment, trajectory, max_samples=10):
        """Get a subset of samples from a specific trajectory"""
        img_paths = self.env_trajectory_map.get((environment, trajectory), [])
        
        if not img_paths:
            return []
            
        # If we need to limit samples, select evenly spaced ones
        if max_samples and len(img_paths) > max_samples:
            indices = np.linspace(0, len(img_paths)-1, max_samples, dtype=int)
            img_paths = [img_paths[i] for i in indices]
            
        # Prepare samples
        samples = []
        for img_path in img_paths:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            
            samples.append({
                'image': image_tensor,
                'path': img_path,
                'environment': environment,
                'trajectory': trajectory,
                'filename': os.path.basename(img_path)
            })
            
        return samples


def load_model(model_path, device='cuda'):
    """Load a trained MobileDepth model from a checkpoint"""
    model = MobileDepth(num_classes=1)
    
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check what's in the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_loss' in checkpoint:
                print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
            if 'abs_rel' in checkpoint:
                print(f"Checkpoint AbsRel: {checkpoint['abs_rel']:.4f}")
            if 'rmse' in checkpoint:
                print(f"Checkpoint RMSE: {checkpoint['rmse']:.4f}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_depth(model, image_tensor, device='cuda'):
    """Generate depth prediction for a single image"""
    try:
        with torch.no_grad():
            # Make sure tensor is on the right device
            input_tensor = image_tensor.unsqueeze(0).to(device)
            torch.cuda.synchronize()  # Synchronize before model inference
            
            # Run model inference
            output = model(input_tensor)
            torch.cuda.synchronize()  # Ensure GPU operations completed
            
            # Handle multi-scale output
            if isinstance(output, list):
                depth_pred = output[0]
            else:
                depth_pred = output
                
            # Move to CPU for post-processing
            depth_pred = depth_pred.squeeze().cpu().numpy()
            
        return depth_pred
        
    except RuntimeError as e:
        print(f"CUDA error in depth prediction: {e}")
        torch.cuda.empty_cache()
        # Return a blank depth map as fallback
        return np.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=np.float32)


def visualize_predictions(images, depth_preds, output_dir, title_prefix="", max_cols=3):
    """Create a visualization grid of multiple predictions"""
    num_samples = len(images)
    num_cols = min(max_cols, num_samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    plt.figure(figsize=(num_cols*5, num_rows*4))
    
    for i, (img, depth) in enumerate(zip(images, depth_preds)):
        # Convert image tensor to numpy
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # RGB image
        ax1 = plt.subplot(num_rows, num_cols*2, i*2+1)
        plt.imshow(img_np)
        plt.title(f"RGB {i+1}")
        plt.axis('off')
        
        # Depth prediction
        ax2 = plt.subplot(num_rows, num_cols*2, i*2+2)
        im = plt.imshow(depth, cmap='plasma')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Depth {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"{title_prefix}_{timestamp}.png")
    
    plt.savefig(output_path, dpi=150)
    plt.close('all')  # Close the figure to free memory
    
    return output_path


def create_depth_video(model, trajectory_samples, output_dir, video_name, fps=10, device='cuda'):
    """Create a video from depth predictions on a trajectory"""
    if not trajectory_samples:
        print("No samples provided for video creation")
        return None
        
    # Get the first sample to determine dimensions
    sample = trajectory_samples[0]
    image_size = sample['image'].shape[1]  # Height of the image
    
    # Set up video writer
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (image_size*2, image_size))
    
    print(f"Creating depth video with {len(trajectory_samples)} frames...")
    
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    for i, sample in enumerate(tqdm(trajectory_samples)):
        # Clear CUDA cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
            
        try:
            # Get RGB image
            img_tensor = sample['image']
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            
            # Generate depth prediction - ensure it's sent to device properly
            with torch.no_grad():
                depth_pred = predict_depth(model, img_tensor, device)
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            
            # Normalize and colorize depth map
            depth_norm = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min() + 1e-8)
            depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            
            # Combine RGB and depth side by side
            combined_frame = np.hstack((img_np, depth_colormap))
            
            # Write frame to video
            video_writer.write(combined_frame)
            
        except RuntimeError as e:
            print(f"Error processing frame {i}: {e}")
            torch.cuda.empty_cache()
            continue
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {video_path}")
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return video_path


def test_full_dataset(model, dataset, device, output_dir, batch_size=1):
    """Test the model on the entire dataset and calculate metrics"""
    # Use smaller batch size and pin_memory=False to avoid CUDA errors
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=False  # Disable pinned memory
    )
    
    # Set up metrics
    metrics_by_env = defaultdict(lambda: {'count': 0, 'time': 0})
    
    print(f"Running model on {len(dataset)} test images...")
    torch.cuda.empty_cache()  # Clear GPU cache before starting
    
    # Process in chunks to avoid memory issues
    max_imgs_per_run = min(500, len(dataset))  # Process at most 500 images per run
    processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Clear CUDA cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
                
            try:
                # Move data to device one by one to avoid misaligned memory
                images = batch['image'].to(device)
                environments = batch['environment']
                
                # Time the prediction
                start_time = time.time()
                outputs = model(images)
                torch.cuda.synchronize()  # Make sure GPU finished execution
                elapsed = time.time() - start_time
                
                # Update metrics for each environment
                for j, env in enumerate(environments):
                    metrics_by_env[env]['count'] += 1
                    metrics_by_env[env]['time'] += elapsed / len(environments)
                
                processed += len(images)
                
                # Break after processing enough images to avoid CUDA errors
                if processed >= max_imgs_per_run:
                    print(f"Processed {processed} images. Stopping to avoid CUDA errors.")
                    break
                    
            except RuntimeError as e:
                print(f"CUDA error encountered: {e}")
                print("Skipping current batch and continuing...")
                torch.cuda.empty_cache()
                continue
    
    # Calculate average inference time per environment
    print("\nInference Time by Environment:")
    for env, metrics in metrics_by_env.items():
        if metrics['count'] > 0:
            avg_time = (metrics['time'] / metrics['count']) * 1000  # Convert to ms
            print(f"  {env}: {avg_time:.2f} ms per image")
    
    # Save metrics to a text file
    metrics_path = os.path.join(output_dir, "inference_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("MobileDepth Model Inference Metrics\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total test images processed: {processed}\n")
        f.write(f"Batch size: {batch_size}\n\n")
        
        f.write("Inference Time by Environment:\n")
        for env, metrics in metrics_by_env.items():
            if metrics['count'] > 0:
                avg_time = (metrics['time'] / metrics['count']) * 1000  # Convert to ms
                f.write(f"  {env}: {avg_time:.2f} ms per image (samples: {metrics['count']})\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    return metrics_by_env


def run_visual_assessment(model, dataset, output_dir, device='cuda'):
    """Run comprehensive visual assessment on the test dataset"""
    # Create output directories
    samples_dir = os.path.join(output_dir, 'sample_predictions')
    videos_dir = os.path.join(output_dir, 'trajectory_videos')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # 1. Generate sample visualizations from each environment
    for env in dataset.environments:
        # Find trajectories for this environment
        env_trajectories = [traj for (environment, traj) in dataset.env_trajectory_map.keys() 
                           if environment == env]
        
        if not env_trajectories:
            print(f"No trajectories found for environment: {env}")
            continue
            
        # Take at most 3 trajectories
        selected_trajectories = env_trajectories[:3]
        
        for traj in selected_trajectories:
            # Get 5 evenly spaced samples from this trajectory
            traj_samples = dataset.get_trajectory_samples(env, traj, max_samples=5)
            
            if not traj_samples:
                continue
                
            # Generate depth predictions
            images = [sample['image'] for sample in traj_samples]
            depth_preds = [predict_depth(model, img, device) for img in images]
            
            # Visualize and save
            title = f"{env}_{traj}_samples"
            visualize_predictions(images, depth_preds, samples_dir, title_prefix=title)
            
            # Create a video for this trajectory (use more frames for video)
            video_samples = dataset.get_trajectory_samples(env, traj, max_samples=30)
            if len(video_samples) >= 10:  # Only create video if we have enough frames
                create_depth_video(model, video_samples, videos_dir, 
                                   video_name=f"{env}_{traj}_depth", device=device)
    
    # 2. Run full dataset inference metrics
    metrics = test_full_dataset(model, dataset, device, output_dir)
    
    # 3. Generate a summary report
    summary_path = os.path.join(output_dir, "visual_assessment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("MobileDepth Visual Assessment Summary\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Dataset type: {dataset.dataset_type}\n")
        f.write(f"  Total test images: {len(dataset)}\n")
        f.write(f"  Environments: {', '.join(dataset.environments)}\n")
        f.write(f"  Trajectories: {len(dataset.env_trajectory_map)}\n\n")
        
        f.write("Visualization Results:\n")
        f.write(f"  Sample predictions: {samples_dir}\n")
        f.write(f"  Trajectory videos: {videos_dir}\n\n")
        
        f.write("Inference Metrics:\n")
        for env, metrics_data in metrics.items():
            avg_time = (metrics_data['time'] / metrics_data['count']) * 1000  # Convert to ms
            f.write(f"  {env}: {avg_time:.2f} ms per image (samples: {metrics_data['count']})\n")
    
    print(f"Assessment summary saved to {summary_path}")


def create_depth_comparison_with_gt(model, root_dir, dataset_type, environment, trajectory, 
                                   output_dir, device='cuda', max_samples=5):
    """Create a visualization comparing predicted depth with ground truth"""
    # In test sets, we might not have ground truth depth maps, but we can use training data for this
    # Change dataset type to training version for comparison
    if dataset_type == 'Kite_test':
        training_dataset = 'Kite_training'
    elif dataset_type == 'PLE_test':
        training_dataset = 'PLE_training'
    else:
        print("Cannot find corresponding training dataset for ground truth comparison")
        return None
    
    # Find color and depth directories
    color_dir = os.path.join(root_dir, training_dataset, environment, 'color_left', trajectory)
    depth_dir = os.path.join(root_dir, training_dataset, environment, 'depth', trajectory)
    
    if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
        print(f"Cannot find color or depth directories for {training_dataset}/{environment}/{trajectory}")
        return None
    
    # Find image pairs
    color_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
        color_files.extend(glob.glob(os.path.join(color_dir, ext)))
    
    color_files = sorted(color_files)
    
    # Select a subset if needed
    if max_samples and len(color_files) > max_samples:
        indices = np.linspace(0, len(color_files)-1, max_samples, dtype=int)
        color_files = [color_files[i] for i in indices]
    
    # Set up transformations
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each pair
    rgb_images = []
    gt_depths = []
    pred_depths = []
    
    for color_file in color_files:
        file_id = os.path.splitext(os.path.basename(color_file))[0]
        depth_file = os.path.join(depth_dir, f"{file_id}.png")
        
        if not os.path.exists(depth_file):
            continue
        
        # Load RGB image
        rgb_img = Image.open(color_file).convert('RGB')
        rgb_tensor = transform(rgb_img)
        
        # Load ground truth depth
        depth_img = Image.open(depth_file)
        depth_img = depth_img.resize((384, 384), Image.NEAREST)
        depth_np = np.array(depth_img).astype(np.float32)
        
        # Normalize ground truth depth
        depth_min, depth_max = depth_np.min(), depth_np.max()
        depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Generate depth prediction
        pred_depth = predict_depth(model, rgb_tensor, device)
        
        # Store all three components
        rgb_images.append(rgb_tensor)
        gt_depths.append(depth_norm)
        pred_depths.append(pred_depth)
    
    if not rgb_images:
        print("No valid image pairs found for comparison")
        return None
    
    # Create comparison visualization
    num_samples = len(rgb_images)
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        # Convert RGB tensor to numpy
        rgb_np = rgb_images[i].permute(1, 2, 0).numpy()
        rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # RGB image
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(rgb_np)
        plt.title(f"RGB {i+1}")
        plt.axis('off')
        
        # Ground truth depth
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(gt_depths[i], cmap='plasma')
        plt.title(f"Ground Truth {i+1}")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # Predicted depth
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(pred_depths[i], cmap='plasma')
        plt.title(f"Prediction {i+1}")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"comparison_{environment}_{trajectory}_{timestamp}.png")
    
    plt.savefig(output_path, dpi=150)
    plt.close('all')  # Close the figure to free memory
    
    print(f"Comparison saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Test MobileDepth model on MidAir test dataset')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='MidAir_dataset',
                        help='Root directory of the MidAir dataset')
    parser.add_argument('--dataset_type', type=str, default='Kite_test',
                        choices=['Kite_test', 'PLE_test', 'VO_test'],
                        help='Test dataset type to use')
    parser.add_argument('--environments', type=str, nargs='+', default=None,
                        help='Specific environments to test on (default: all)')
    
    # Test parameters
    parser.add_argument('--image_size', type=int, default=384,
                        help='Image size for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for quick testing)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for testing (cuda or cpu)')
    parser.add_argument('--with_gt_comparison', action='store_true',
                        help='Generate comparisons with ground truth depth (requires training data)')
    parser.add_argument('--safe_mode', action='store_true',
                        help='Run in safe mode with reduced memory usage to avoid CUDA errors')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print system information
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {args.device}")
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        # Initial CUDA memory cleanup
        torch.cuda.empty_cache()
    
    # Load model
    model = load_model(args.model_path, device=args.device)
    torch.cuda.empty_cache()  # Clear memory after model loading
    
    # Initialize dataset with fewer samples if in safe mode
    if args.safe_mode and args.max_samples is None:
        args.max_samples = 50
        print(f"Running in safe mode with max_samples={args.max_samples}")
    
    test_dataset = TestDataset(
        root_dir=args.data_root,
        dataset_type=args.dataset_type,
        environments=args.environments,
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    # Run comprehensive visual assessment
    run_visual_assessment(model, test_dataset, args.output_dir, device=args.device)
    
    # Clean up before ground truth comparison
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Generate ground truth comparisons if requested
    if args.with_gt_comparison:
        comparison_dir = os.path.join(args.output_dir, 'gt_comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        print("\nGenerating ground truth comparisons...")
        for (env, traj) in test_dataset.env_trajectory_map.keys():
            # Try to find corresponding training trajectory - may not exist
            # We'll just use the same trajectory ID for simplicity
            create_depth_comparison_with_gt(
                model, args.data_root, args.dataset_type, env, traj,
                comparison_dir, device=args.device, 
                max_samples=3 if args.safe_mode else 5
            )
            
            # Clear GPU memory after each comparison
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    
    print("\nTest completed! Results saved to:", os.path.abspath(args.output_dir))
    
    # Final cleanup
    if args.device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
