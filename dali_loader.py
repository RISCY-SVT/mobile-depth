import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Check if DALI is available and import
try:
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("NVIDIA DALI not found. Install with: pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110")

class DALIDataLoader:
    """DALI-based data loader for depth estimation"""
    
    def __init__(self, root_dir, datasets, environments, batch_size=12, image_size=384, 
                 is_train=True, val_split=0.1, num_workers=4):
        
        if not DALI_AVAILABLE:
            raise ImportError("NVIDIA DALI is required but not installed")
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        
        # Collect file paths
        self.rgb_files = []
        self.depth_files = []
        
        # Find all RGB-Depth pairs
        for dataset in datasets:
            for env in environments:
                env_path = os.path.join(root_dir, dataset, env)
                if not os.path.exists(env_path):
                    continue
                
                color_dir = os.path.join(env_path, 'color_left')
                depth_dir = os.path.join(env_path, 'depth')
                
                if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                    continue
                
                # Find files (similar to other loaders)
                # ... (file collection code) ...
        
        # Split into train/val
        # ... (split code) ...
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        self.pipeline.build()
        
        # Create DALI iterator
        self.dali_iter = DALIGenericIterator(
            [self.pipeline], 
            ['rgb', 'depth'],
            size=len(self.rgb_files) // batch_size,
            auto_reset=True
        )
    
    @pipeline_def
    def _create_pipeline(self):
        # File readers
        rgb_files, rgb_labels = fn.readers.file(
            files=self.rgb_files,
            random_shuffle=self.is_train
        )
        
        depth_files, depth_labels = fn.readers.file(
            files=self.depth_files,
            random_shuffle=self.is_train
        )
        
        # Decode images
        rgb = fn.decoders.image(
            rgb_files,
            device="mixed",
            output_type=types.RGB
        )
        
        depth = fn.decoders.image(
            depth_files,
            device="mixed",
            output_type=types.GRAY
        )
        
        # Resize images
        rgb = fn.resize(
            rgb,
            size=[self.image_size, self.image_size]
        )
        
        depth = fn.resize(
            depth,
            size=[self.image_size, self.image_size],
            interp_type=types.INTERP_NN
        )
        
        # Data augmentation (if training)
        if self.is_train:
            # Random horizontal flip
            flip = fn.random.coin_flip(probability=0.5)
            rgb = fn.flip(rgb, horizontal=flip)
            depth = fn.flip(depth, horizontal=flip)
            
            # Color jitter
            rgb = fn.brightness_contrast(rgb, brightness=0.2, contrast=0.2)
        
        # Normalize RGB
        rgb = fn.normalize(
            rgb,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        
        # Normalize depth (per-image min-max)
        depth_min = fn.reductions.min(depth)
        depth_max = fn.reductions.max(depth)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Transpose for PyTorch (NCHW)
        rgb = fn.transpose(rgb, perm=[2, 0, 1])
        
        return rgb, depth
    
    def __iter__(self):
        return self.dali_iter
    
    def __len__(self):
        return len(self.dali_iter)
