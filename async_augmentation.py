import os
import torch
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
import queue
import threading
import time
import random
from PIL import Image
import numpy as np
from torchvision import transforms

class AsyncAugmentationDataset(Dataset):
    """Dataset with asynchronous data augmentation"""
    
    def __init__(self, root_dir, datasets, environments, image_size=384, is_train=True, 
                 val_split=0.1, num_workers=4, queue_size=100):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # Collect file paths
        self.rgb_files = []
        self.depth_files = []
        
        # Find all RGB-Depth pairs
        # ... (same file collection code) ...
        
        # Split train/val
        # ... (same split code) ...
        
        # Base transforms (no augmentation)
        self.rgb_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Augmentation transforms (only for training)
        self.augmentation_enabled = is_train
        
        # Setup async augmentation
        self.augmentation_queue = queue.Queue(maxsize=queue_size)
        self.augmentation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.stop_event = threading.Event()
        
        # Start background workers
        self.prefetch_thread = threading.Thread(target=self._async_prefetch)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
    
    def _apply_augmentation(self, rgb, depth):
        """Apply augmentations to the rgb and depth pair"""
        # Only augment during training
        if not self.augmentation_enabled:
            return rgb, depth
        
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = torch.flip(rgb, [2])  # Flip width dimension
            depth = torch.flip(depth, [2])
        
        # Brightness and contrast (RGB only)
        if random.random() > 0.7:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            
            # Unnormalize, apply adjustment, renormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_denorm = rgb * std + mean
            
            # Apply adjustments
            rgb_denorm = torch.clamp(rgb_denorm * brightness, 0, 1)
            gray = rgb_denorm.mean(dim=0, keepdim=True)
            rgb_denorm = torch.clamp((rgb_denorm - gray) * contrast + gray, 0, 1)
            
            # Renormalize
            rgb = (rgb_denorm - mean) / std
        
        return rgb, depth
    
    def _load_and_process(self, idx):
        """Load and process a single item"""
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        
        # Load RGB
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.rgb_transform_base(rgb_img)
        
        # Load depth
        depth_img = Image.open(depth_path)
        depth_tensor = self.depth_transform_base(depth_img)
        
        # Normalize depth with min-max
        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
        depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Ensure single channel
        if depth_tensor.shape[0] > 1:
            depth_tensor = depth_tensor[0].unsqueeze(0)
        
        # Apply augmentation asynchronously if enabled
        if self.augmentation_enabled:
            rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'rgb_path': rgb_path,
            'depth_path': depth_path
        }
    
    def _async_prefetch(self):
        """Background thread to prefetch and augment data"""
        try:
            indices = list(range(len(self.rgb_files)))
            while not self.stop_event.is_set():
                random.shuffle(indices)
                
                # Submit batch of tasks
                future_to_idx = {}
                for idx in indices:
                    if self.stop_event.is_set():
                        break
                    
                    future = self.augmentation_pool.submit(self._load_and_process, idx)
                    future_to_idx[future] = idx
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    if self.stop_event.is_set():
                        break
                        
                    idx = future_to_idx[future]
                    try:
                        item = future.result()
                        # Try to put in queue, but don't block if full
                        try:
                            self.augmentation_queue.put((idx, item), block=True, timeout=1.0)
                        except queue.Full:
                            pass
                    except Exception as e:
                        print(f"Error processing item {idx}: {e}")
        except Exception as e:
            print(f"Async prefetch error: {e}")
            torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Try to get from queue first
        try:
            # Non-blocking check with small timeout
            queue_item = self.augmentation_queue.get(block=True, timeout=0.01)
            if queue_item[0] == idx:
                # We got lucky! Return this item
                return queue_item[1]
            else:
                # Put it back for someone else
                self.augmentation_queue.put(queue_item, block=False)
        except (queue.Empty, queue.Full):
            pass
        
        # Fall back to synchronous loading if not in queue
        return self._load_and_process(idx)
    
    def cleanup(self):
        """Cleanup background threads"""
        self.stop_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
        self.augmentation_pool.shutdown(wait=False)
