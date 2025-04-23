import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import argparse
import time
import torch
from bdepth_model import B_MobileDepth  # Изменен импорт на новую модель

# Import optimized components
from improved_fix_data_loader import EnhancedRAMCachedDataset, create_improved_dataloader
from optimized_data_loader import RAMCachedDataset
from improved_training_loop import train_improved

# torch.backends.cudnn.enabled = False

# Настраиваем выделение памяти
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def main():
    torch.backends.cudnn.benchmark = False      # Выключаем benchmark для более предсказуемого поведения
    torch.backends.cudnn.deterministic = True   # Включаем детерминированные алгоритмы
    torch.cuda.empty_cache()                    # Очищаем кэш CUDA для предотвращения утечек памяти
    torch.cuda.reset_peak_memory_stats()        # Сбрасываем статистику пикового использования памяти
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized training for B_MobileDepth model on MidAir dataset')
    
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
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, 
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, 
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                      help='Weight decay for optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                      help='Gradient clipping value')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                      help='Minimum learning rate')
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
    parser.add_argument('--num_workers', type=int, default=10, 
                      help='Number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=4, 
                      help='Number of batches to prefetch per worker')
    parser.add_argument('--cache_size', type=int, default=50000, 
                      help='Number of images to cache in RAM')
    parser.add_argument('--pin_memory', action='store_true', default=True, 
                      help='Pin memory for faster GPU transfer')
    parser.add_argument('--disable_cache', action='store_true', 
                      help='Disable RAM caching')
    parser.add_argument('--disable_augmentation', action='store_true', 
                      help='Disable data augmentation')
    
    # Augmentation parameters for B_MobileDepth
    parser.add_argument('--strong_augmentation', action='store_true', 
                      help='Enable stronger data augmentation')
    parser.add_argument('--noise_augmentation', action='store_true', 
                      help='Add noise augmentation for better quantization')
    parser.add_argument('--perspective_augmentation', action='store_true', 
                      help='Add perspective transform augmentation')
    
    # Quantization parameters
    parser.add_argument('--use_qat', action='store_true', 
                      help='Use Quantization-Aware Training')
    parser.add_argument('--test_quantization', action='store_true', 
                      help='Test model with simulated quantization')
    parser.add_argument('--use_robust_loss', action='store_true', 
                      help='Use more robust loss function')
    
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
    print(f"Strong augmentation: {'Enabled' if args.strong_augmentation else 'Disabled'}")
    print(f"Noise augmentation: {'Enabled' if args.noise_augmentation else 'Disabled'}")
    print(f"Perspective augmentation: {'Enabled' if args.perspective_augmentation else 'Disabled'}")
    print(f"Quantization-Aware Training: {'Enabled' if args.use_qat else 'Disabled'}")
    print(f"Robust loss function: {'Enabled' if args.use_robust_loss else 'Disabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize datasets
    print("\n=== Loading Datasets ===")
    train_dataset = EnhancedRAMCachedDataset(
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
        strong_augmentation=args.strong_augmentation,
        noise_augmentation=args.noise_augmentation,
        perspective_augmentation=args.perspective_augmentation,
        debug=args.debug
    )

    val_dataset = EnhancedRAMCachedDataset(
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
    print("\n=== Initializing B_MobileDepth Model ===")
    model = B_MobileDepth(input_size=(args.image_size, args.image_size))
    
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
    model = train_improved(args, model, train_dataset, val_dataset)
    
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
