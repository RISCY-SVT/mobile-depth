import os
import glob
import random
import time
import threading
import queue
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import psutil
import gc

class RAMCachedDataset(Dataset):
    """
    Dataset with RAM caching for extremely fast data loading on machines with high RAM.
    Optimized for systems with large memory but slower disk access.
    """
    def __init__(self, root_dir, datasets, environments, image_size=640, is_train=True, 
                 val_split=0.1, depth_scale=100.0, max_total_images=None, 
                 max_imgs_per_trajectory=None, cache_size=20000, enable_cache=True,
                 enable_augmentation=True, debug=False):
        """
        Initialize the dataset with RAM caching capabilities.
        
        Args:
            root_dir: Root directory of the dataset
            datasets: List of dataset types to include (e.g., ['Kite_training'])
            environments: List of environments to include (e.g., ['sunny', 'foggy'])
            image_size: Size to resize images to
            is_train: Whether this is for training or validation
            val_split: Fraction of data to use for validation
            depth_scale: Scale factor for depth maps
            max_total_images: Maximum number of images to use (for quick testing)
            max_imgs_per_trajectory: Maximum images to use from each trajectory
            cache_size: Maximum number of images to keep in RAM cache
            enable_cache: Whether to enable the RAM cache
            enable_augmentation: Whether to apply data augmentation
            debug: Enable debug output
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.enable_cache = enable_cache
        self.enable_augmentation = enable_augmentation and is_train  # Only augment training data
        self.debug = debug
        self.is_train = is_train
        
        # Set up RAM cache using OrderedDict for LRU behavior
        self.cache_size = cache_size if enable_cache else 0
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cache_report_time = time.time()
        
        # Track memory usage
        self.memory_usage = []
        
        # Initialize data paths
        self.rgb_files = []
        self.depth_files = []
        
        # Track statistics
        self.trajectories_processed = 0
        self.start_time = time.time()
        
        print(f"Initializing {'training' if is_train else 'validation'} dataset with RAM caching")
        print(f"  - Cache size: {self.cache_size} images")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Available system RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Find all image pairs
        total_pairs_found = 0
        trajectories_with_limits = defaultdict(int)
        
        for dataset in datasets:
            for env in environments:
                env_path = os.path.join(root_dir, dataset, env)
                if not os.path.exists(env_path):
                    print(f"Skipping {env_path}: directory doesn't exist")
                    continue
                
                color_dir = os.path.join(env_path, 'color_left')
                depth_dir = os.path.join(env_path, 'depth')
                
                if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                    print(f"Skipping {env_path}: missing color_left or depth directory")
                    continue
                
                print(f"Processing {dataset}/{env}")
                
                trajectory_pairs = 0
                for trajectory_dir in os.listdir(color_dir):
                    rgb_trajectory_path = os.path.join(color_dir, trajectory_dir)
                    depth_trajectory_path = os.path.join(depth_dir, trajectory_dir)
                    
                    if not os.path.isdir(rgb_trajectory_path) or not os.path.exists(depth_trajectory_path):
                        continue
                    
                    # Find all image files with different possible extensions
                    rgb_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
                        rgb_files.extend(glob.glob(os.path.join(rgb_trajectory_path, ext)))
                    
                    rgb_files = sorted(rgb_files)
                    
                    # Apply trajectory image limit if specified
                    if max_imgs_per_trajectory and len(rgb_files) > max_imgs_per_trajectory:
                        # Use evenly spaced samples throughout the trajectory
                        indices = np.linspace(0, len(rgb_files)-1, max_imgs_per_trajectory, dtype=int)
                        rgb_files = [rgb_files[i] for i in indices]
                        trajectories_with_limits[trajectory_dir] = len(indices)
                    
                    pairs_in_trajectory = 0
                    for rgb_file in rgb_files:
                        file_id = os.path.splitext(os.path.basename(rgb_file))[0]
                        depth_file = os.path.join(depth_trajectory_path, f"{file_id}.png")
                        
                        if os.path.exists(depth_file):
                            self.rgb_files.append(rgb_file)
                            self.depth_files.append(depth_file)
                            pairs_in_trajectory += 1
                    
                    if pairs_in_trajectory > 0:
                        trajectory_pairs += pairs_in_trajectory
                        self.trajectories_processed += 1
                        if self.debug:
                            print(f"  Trajectory {trajectory_dir}: {pairs_in_trajectory} pairs")
                
                total_pairs_found += trajectory_pairs
                print(f"  Found {trajectory_pairs} pairs in {dataset}/{env}")
        
        if trajectories_with_limits:
            print(f"Applied trajectory limits to {len(trajectories_with_limits)} trajectories")
        
        print(f"Total pairs found: {total_pairs_found}")
        
        # Apply global image limit if specified
        if max_total_images and total_pairs_found > max_total_images:
            random.seed(42)
            indices = random.sample(range(total_pairs_found), max_total_images)
            self.rgb_files = [self.rgb_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]
            print(f"Randomly sampled {max_total_images} pairs from total of {total_pairs_found}")
        
        # Split into train/val
        if total_pairs_found > 0:
            all_indices = list(range(len(self.rgb_files)))
            random.seed(42)
            random.shuffle(all_indices)
            
            split_idx = int(val_split * len(all_indices))
            selected_indices = all_indices[split_idx:] if is_train else all_indices[:split_idx]
            
            self.rgb_files = [self.rgb_files[i] for i in selected_indices]
            self.depth_files = [self.depth_files[i] for i in selected_indices]
            
            print(f"{'Training' if is_train else 'Validation'} examples: {len(self.rgb_files)}")
        else:
            print("WARNING: No valid image pairs found!")
        
        # Base transformations (always applied)
        self.rgb_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Data augmentation transformations (only applied during training)
        if self.enable_augmentation:
            print("Data augmentation enabled")
        
        # Initialize prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=100)
        self.prefetch_idx = 0
        self.prefetch_running = False
        self.prefetch_thread = None
        
        # Start prefetching if cache is enabled
        if self.enable_cache and len(self.rgb_files) > 0:
            self._start_prefetching()
    
    def _start_prefetching(self):
        """Start a background thread to prefetch and cache data"""
        if self.prefetch_running:
            return
            
        self.prefetch_running = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        print("Started background prefetching thread")
    
    def _prefetch_worker(self):
        """Worker thread to prefetch data into RAM cache"""
        print("Prefetch worker started")
        try:
            indices = list(range(len(self.rgb_files)))
            random.shuffle(indices)  # Prefetch in random order for better cache distribution
            
            for idx in indices:
                if not self.prefetch_running:
                    break
                    
                # Skip if already in cache
                if idx in self.cache:
                    continue
                
                # Check memory usage and clear cache if needed
                if len(self.cache) >= self.cache_size:
                    continue  # Let the LRU mechanism handle cache eviction
                
                # Load the data
                try:
                    rgb_path = self.rgb_files[idx]
                    depth_path = self.depth_files[idx]
                    
                    # Load and preprocess RGB image
                    rgb_image = Image.open(rgb_path).convert('RGB')
                    rgb_tensor = self.rgb_transform_base(rgb_image)
                    
                    # Load and preprocess depth image
                    depth_image = Image.open(depth_path)
                    depth_tensor = self.depth_transform_base(depth_image)
                    
                    # Normalize depth
                    # depth_tensor = torch.clamp(depth_tensor / self.depth_scale, 0, 1)
                    # Alternative approach
                    depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
                    depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)  # [0,1] range
                    
                    # Ensure single-channel depth
                    if depth_tensor.shape[0] > 1:
                        depth_tensor = depth_tensor[0].unsqueeze(0)
                    
                    # Store in cache
                    self.cache[idx] = (rgb_tensor, depth_tensor)
                    
                    # Track memory usage occasionally
                    if random.random() < 0.01:  # 1% chance to measure memory
                        self.memory_usage.append(psutil.virtual_memory().percent)
                        
                        # Report cache stats periodically
                        current_time = time.time()
                        if current_time - self.last_cache_report_time > 60:  # Report every minute
                            self._report_cache_stats()
                            self.last_cache_report_time = current_time
                            
                except Exception as e:
                    print(f"Error prefetching index {idx}: {e}")
                
                # Free up some CPU time
                time.sleep(0.001)
        
        except Exception as e:
            print(f"Prefetch worker exception: {e}")
        
        finally:
            self.prefetch_running = False
    
    def _report_cache_stats(self):
        """Report cache statistics"""
        cache_size_mb = sum(t.element_size() * t.nelement() for pair in self.cache.values() for t in pair) / (1024**2)
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests) * 100
        
        print(f"Cache stats: {len(self.cache)}/{self.cache_size} entries, {cache_size_mb:.1f} MB, "
              f"Hit rate: {hit_rate:.1f}% ({self.cache_hits}/{total_requests})")
        print(f"Memory usage: {psutil.virtual_memory().percent}%, "
              f"Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _apply_augmentation(self, rgb, depth):
        """Apply data augmentation to the RGB and depth pair"""
        if not self.enable_augmentation:
            return rgb, depth
        
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            rgb = torch.flip(rgb, [2])  # Flip horizontally
            depth = torch.flip(depth, [2])  # Flip horizontally
        
        # Random brightness and contrast adjustment to RGB (30% chance)
        if random.random() > 0.7:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            # Denormalize
            means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_denorm = rgb * stds + means
            
            # Adjust brightness and contrast
            rgb_denorm = torch.clamp(rgb_denorm * brightness_factor, 0, 1)
            gray = rgb_denorm.mean(dim=0, keepdim=True)
            rgb_denorm = torch.clamp((rgb_denorm - gray) * contrast_factor + gray, 0, 1)
            
            # Renormalize
            rgb = (rgb_denorm - means) / stds
        
        return rgb, depth
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Try to get from cache first
        if self.enable_cache and idx in self.cache:
            self.cache_hits += 1
            # Move the item to the end of the OrderedDict (most recently used)
            rgb_tensor, depth_tensor = self.cache.pop(idx)
            self.cache[idx] = (rgb_tensor, depth_tensor)
            
            # Apply augmentation
            if self.enable_augmentation:
                rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
                
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': self.rgb_files[idx],
                'depth_path': self.depth_files[idx],
                'from_cache': True
            }
        
        # Cache miss - load from disk
        self.cache_misses += 1
        
        # Load RGB image
        rgb_path = self.rgb_files[idx]
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.rgb_transform_base(rgb_image)
        
        # Load depth image
        depth_path = self.depth_files[idx]
        depth_image = Image.open(depth_path)
        depth_tensor = self.depth_transform_base(depth_image)
        
        # Normalize depth
        # depth_tensor = torch.clamp(depth_tensor / self.depth_scale, 0, 1)
        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
        depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)  # [0,1] range
        
        # Ensure single-channel depth
        if depth_tensor.shape[0] > 1:
            depth_tensor = depth_tensor[0].unsqueeze(0)
        
        # Apply augmentation if enabled
        if self.enable_augmentation:
            rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
        
        # Add to cache if enabled
        if self.enable_cache:
            # If cache is full, remove the oldest item (first in OrderedDict)
            if len(self.cache) >= self.cache_size:
                self.cache.popitem()
            
            # Add current item to cache
            self.cache[idx] = (rgb_tensor.clone(), depth_tensor.clone())
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'from_cache': False
        }
    
    def shutdown(self):
        """Shutdown the prefetching thread"""
        self.prefetch_running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
        self._report_cache_stats()
