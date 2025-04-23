import os
import torch
import random
import numpy as np
import threading
import time
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from optimized_data_loader import RAMCachedDataset

# Define seed worker at module level so it can be pickled
def seed_worker(worker_id):
    # Set random seeds for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class EnhancedRAMCachedDataset(RAMCachedDataset):
    """
    Improved caching dataset with support for advanced augmentations
    and stabilization for quantization.
    """
    def __init__(self, *args, **kwargs):
        # Extract necessary parameters before passing to parent class
        self.strong_augmentation = kwargs.pop('strong_augmentation', False)
        self.noise_augmentation = kwargs.pop('noise_augmentation', False)
        self.perspective_augmentation = kwargs.pop('perspective_augmentation', False)
        self.normalize_depth = kwargs.pop('normalize_depth', True)
        
        super().__init__(*args, **kwargs)
        
        # Immediately stop the prefetching thread
        self.prefetch_running = False
        if hasattr(self, 'prefetch_thread') and self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
            self.prefetch_thread = None
        
        # Clear cache for pickling
        self.cache = {}
        
        # Set up additional augmentations for training
        if self.enable_augmentation:
            self._setup_enhanced_augmentations()
    
    def _setup_enhanced_augmentations(self):
        """Set up enhanced augmentations for better generalization capabilities
        and quantization robustness"""
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2 if self.strong_augmentation else 0.1,
            contrast=0.2 if self.strong_augmentation else 0.1,
            saturation=0.2 if self.strong_augmentation else 0.1,
            hue=0.1 if self.strong_augmentation else 0.05
        )
        
        # Perspective and affine transform augmentations
        if self.perspective_augmentation:
            self.perspective_transform = transforms.RandomPerspective(
                distortion_scale=0.1 if self.strong_augmentation else 0.05, 
                p=0.3
            )
            self.affine_transform = transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=2
            )
    
    def __getstate__(self):
        """Custom getstate to handle pickling"""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state['cache'] = OrderedDict()
        state['prefetch_thread'] = None
        state['prefetch_queue'] = None
        state['memory_usage'] = []
        return state
    
    def __setstate__(self, state):
        """Custom setstate to handle unpickling"""
        self.__dict__.update(state)
        # Re-initialize but don't start prefetching
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefetch_running = False
        self.prefetch_thread = None
        self.prefetch_queue = None
        self.memory_usage = []
        
        # Re-setup augmentations if needed
        if hasattr(self, 'enable_augmentation') and self.enable_augmentation:
            self._setup_enhanced_augmentations()
    
    def _apply_augmentation(self, rgb, depth):
        """Apply data augmentation to the RGB and depth pair, with improved
        handling for quantization robustness"""
        if not self.enable_augmentation:
            return rgb, depth
        
        # Ensure same random state for consistency between RGB and depth
        seed = torch.randint(0, 2147483647, (1,)).item()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            rgb = torch.flip(rgb, [2])  # Flip horizontally
            depth = torch.flip(depth, [2])  # Flip horizontally
        
        # Advanced color augmentations for RGB image
        if random.random() > 0.5:
            # Convert to PIL for color jitter
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            rgb_pil = transforms.ToPILImage()(rgb_np)
            
            # Apply color jitter
            rgb_pil = self.color_jitter(rgb_pil)
            
            # Convert back to tensor
            rgb = transforms.ToTensor()(rgb_pil)
            
            # Renormalize
            rgb = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(rgb)
        
        # Perspective and affine transformations (apply to RGB and depth equally)
        if self.perspective_augmentation and random.random() > 0.7:
            # Convert RGB to PIL format for transformations
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            rgb_np = np.clip(rgb_np, 0, 1)  # Normalize to range [0, 1]
            rgb_pil = transforms.ToPILImage()(rgb_np)
            
            # Convert depth to PIL format
            depth_np = depth.permute(1, 2, 0).cpu().numpy()
            depth_np = np.clip(depth_np, 0, 1)
            depth_pil = transforms.ToPILImage()(depth_np)
            
            # Apply the same transformation to both images
            if random.random() > 0.5:
                # Perspective transform
                rgb_pil = self.perspective_transform(rgb_pil)
                depth_pil = self.perspective_transform(depth_pil)
            else:
                # Affine transform
                rgb_pil = self.affine_transform(rgb_pil)
                depth_pil = self.affine_transform(depth_pil)
            
            # Convert back to tensors
            rgb = transforms.ToTensor()(rgb_pil)
            rgb = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(rgb)
            
            depth = transforms.ToTensor()(depth_pil)
        
        # Add slight noise to RGB and depth for better quantization robustness
        if self.noise_augmentation and random.random() > 0.7:
            # Add noise to RGB (smaller magnitude)
            rgb_noise = torch.randn_like(rgb) * 0.01
            rgb = torch.clamp(rgb + rgb_noise, -2.0, 2.0)  # Clamp to reasonable range considering normalization
            
            # Add noise to depth (calibrated to be subtle)
            depth_noise = torch.randn_like(depth) * 0.005
            depth = torch.clamp(depth + depth_noise, 0.001, 0.999)  # Ensure valid depth values
        
        return rgb, depth
    
    def __getitem__(self, idx):
        """Override __getitem__ with improved augmentations and stabilization for quantization"""
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
        rgb_image = self.rgb_transform_base(self._load_and_resize_image(rgb_path))
        
        # Load depth image
        depth_path = self.depth_files[idx]
        depth_image = self.depth_transform_base(self._load_and_resize_image(depth_path, is_depth=True))
        
        # Normalize depth with protection against zero values for better quantization
        if self.normalize_depth:
            depth_min, depth_max = depth_image.min(), depth_image.max()
            if depth_max > depth_min:  # Check for degenerate case
                depth_image = (depth_image - depth_min) / (depth_max - depth_min + 1e-8)
            depth_image = torch.clamp(depth_image, 0.001, 0.999)  # Prevent extreme values
        
        # Ensure single-channel depth
        if depth_image.shape[0] > 1:
            depth_image = depth_image[0].unsqueeze(0)
        
        # Apply augmentation if enabled
        if self.enable_augmentation:
            rgb_image, depth_image = self._apply_augmentation(rgb_image, depth_image)
        
        # Add to cache if enabled
        if self.enable_cache:
            # If cache is full, remove the oldest item (first in OrderedDict)
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove oldest item
            
            # Add current item to cache
            self.cache[idx] = (rgb_image.clone(), depth_image.clone())
        
        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'from_cache': False
        }
    
    def _load_and_resize_image(self, path, is_depth=False):
        """Optimized image loading and resizing with error handling"""
        import PIL.Image as pil_image
        try:
            img = pil_image.open(path)
            if not is_depth:
                img = img.convert('RGB')
            return img.resize((self.image_size, self.image_size), 
                            resample=pil_image.NEAREST if is_depth else pil_image.BILINEAR)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a dummy image in case of error
            mode = 'L' if is_depth else 'RGB'
            return pil_image.new(mode, (self.image_size, self.image_size))

def create_improved_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=0, 
                           persistent_workers=False, drop_last=False, worker_init_fn=None,
                           prefetch_factor=2):
    r"""
    Creates an optimized data loader with settings for best
    performance and stability when training models for quantization.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        pin_memory: Whether to use pinned memory for faster GPU transfer
        num_workers: Number of worker processes (0 = single main thread)
        persistent_workers: Whether to keep worker processes between epochs
        drop_last: Whether to drop the last incomplete batch
        worker_init_fn: Worker initialization function
        prefetch_factor: Data prefetch multiplier
    """
    
    # Use the module-level seed_worker if no worker_init_fn is provided
    if worker_init_fn is None:
        worker_init_fn = seed_worker
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return loader

# ─── qat_prefetcher.py ──────────────────────────────────────────────────────────
# qat_prefetcher.py
class QATPrefetcher:
    """
    Prefetcher без двойного GPU-буфера:
    ─ читает следующий batch в  CPU-RAM;
    ─ при  .next()   синхронно копирует  CPU→GPU  только этот batch;
      на GPU одновременно живёт ровно один batch.
    """

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 device: torch.device | str = "cuda"):
        self._base_loader = loader            # сохраняем для reset()
        self.device = device
        self._loader_iter = iter(loader)      # собственный итератор
        self._next_batch_cpu = None           # хранится в RAM
        self._preload()                       # загружаем первый batch (CPU)

    # ──────────────────────────────────────────────────────────────────────────
    def _preload(self) -> None:
        """Читаем следующий batch в CPU-память (без копий на GPU)."""
        try:
            self._next_batch_cpu = next(self._loader_iter)
        except StopIteration:
            self._next_batch_cpu = None       # эпоха закончилась

    # ──────────────────────────────────────────────────────────────────────────
    def next(self) -> dict | None:
        """
        Возвращает dict c тенсорами **уже на GPU**.
        На GPU к моменту return будет находиться **только** этот batch.
        """
        if self._next_batch_cpu is None:      # конец эпохи
            return None

        # 1) берём CPU-batch, запускаем асинхронную загрузку следующего
        batch_cpu = self._next_batch_cpu
        self._preload()                       # читает следующий batch (CPU)

        # 2) копируем текущий batch на GPU (синхронно с compute-stream’ом)
        batch_gpu = {}
        for k, v in batch_cpu.items():
            if torch.is_tensor(v):
                batch_gpu[k] = v.to(self.device, non_blocking=True)
            else:
                batch_gpu[k] = v

        return batch_gpu

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Начать эпоху заново (нужен новый итератор DataLoader)."""
        self._loader_iter = iter(self._base_loader)
        self._next_batch_cpu = None
        self._preload()
