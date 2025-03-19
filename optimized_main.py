import os
import argparse
import time
import torch
from depth_model import MobileDepth

# Import optimized components
from fix_data_loader import FixedRAMCachedDataset, create_single_process_dataloader
from optimized_data_loader import RAMCachedDataset
from optimized_training_loop import train_optimized

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized training for MobileDepth model on MidAir dataset')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='MidAir_dataset', help='Root directory of the dataset')
    parser.add_argument('--datasets', type=str, nargs='+', default=['Kite_training'], 
                      help='Dataset types to use (Kite_training, PLE_training)')
    parser.add_argument('--environments', type=str, nargs='+', default=['sunny', 'foggy', 'cloudy', 'sunset'], 
                      help='Environments to include (sunny, foggy, cloudy, sunset, fall, spring, winter)')
    parser.add_argument('--max_total_images', type=int, default=None, 
                      help='Maximum total number of images (None = use all)')
    parser.add_argument('--max_imgs_per_trajectory', type=int, default=None, 
                      help='Maximum images per trajectory (None = use all)')
    parser.add_argument('--image_size', type=int, default=320, 
                      help='Size to resize images (smaller = faster training)')
    parser.add_argument('--depth_scale', type=float, default=100.0, 
                      help='Scale factor for depth maps')
    parser.add_argument('--val_split', type=float, default=0.1, 
                      help='Fraction of data to use for validation')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=12, 
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, 
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='Initial learning rate')
    parser.add_argument('--val_freq', type=int, default=1, 
                      help='Validate every N epochs')
    parser.add_argument('--save_freq', type=int, default=1, 
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, 
                      help='Path to checkpoint to resume from')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', 
                      help='Directory to save outputs')
    
    # Performance parameters
    parser.add_argument('--mixed_precision', action='store_true', 
                      help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=16, 
                      help='Number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=4, 
                      help='Number of batches to prefetch per worker')
    parser.add_argument('--cache_size', type=int, default=20000, 
                      help='Number of images to cache in RAM')
    parser.add_argument('--pin_memory', action='store_true', default=True, 
                      help='Pin memory for faster GPU transfer')
    parser.add_argument('--disable_cache', action='store_true', 
                      help='Disable RAM caching')
    parser.add_argument('--disable_augmentation', action='store_true', 
                      help='Disable data augmentation')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true', 
                      help='Enable debug output')
    parser.add_argument('--profile', action='store_true', 
                      help='Enable profiling')
    
    args = parser.parse_args()
    
    # Print system information
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    import psutil
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Time the script execution
    script_start_time = time.time()
    
    # Print the optimization settings
    print("\n=== Optimization Settings ===")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"RAM cache: {'Disabled' if args.disable_cache else f'Enabled, size: {args.cache_size}'}")
    print(f"Data augmentation: {'Disabled' if args.disable_augmentation else 'Enabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize datasets
    print("\n=== Loading Datasets ===")
    train_dataset = FixedRAMCachedDataset(
        root_dir=args.data_root,
        datasets=args.datasets,
        environments=args.environments,
        image_size=args.image_size,
        is_train=True,
        val_split=args.val_split,
        depth_scale=args.depth_scale,
        max_total_images=args.max_total_images,
        max_imgs_per_trajectory=args.max_imgs_per_trajectory,
        cache_size=args.cache_size,
        enable_cache=not args.disable_cache,
        enable_augmentation=not args.disable_augmentation,
        debug=args.debug
    )

    val_dataset = FixedRAMCachedDataset(
        root_dir=args.data_root,
        datasets=args.datasets,
        environments=args.environments,
        image_size=args.image_size,
        is_train=False,
        val_split=args.val_split,
        depth_scale=args.depth_scale,
        max_total_images=None,  # Use all validation data
        max_imgs_per_trajectory=args.max_imgs_per_trajectory,
        cache_size=min(args.cache_size // 5, 5000),  # Smaller cache for validation
        enable_cache=not args.disable_cache,
        enable_augmentation=False,  # No augmentation for validation
        debug=args.debug
    )
    
    # Create model
    print("\n=== Initializing Model ===")
    model = MobileDepth(num_classes=1)
    
    # Get model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Profiling if requested
    if args.profile:
        try:
            from torch.profiler import profile, record_function, ProfilerActivity
            print("Profiling enabled - will generate profiling report after first epoch")
        except ImportError:
            print("Profiling requested but torch.profiler not available")
            args.profile = False
    
    # Train the model
    print("\n=== Starting Training ===")
    model = train_optimized(args, model, train_dataset, val_dataset)
    
    # Final cleanup and timing
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n=== Training Complete ===")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
