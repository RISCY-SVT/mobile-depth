import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import random
import tempfile
from torchvision import transforms

class MmapDataset(Dataset):
    """Dataset that creates and uses memory-mapped arrays for faster loading"""
    
    def __init__(self, root_dir, datasets, environments, image_size=384, is_train=True, 
                 val_split=0.1, debug=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        self.debug = debug
        
        # Collect all image paths first
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
                
                # For each trajectory directory, find matching images
                for traj in os.listdir(color_dir):
                    color_traj = os.path.join(color_dir, traj)
                    depth_traj = os.path.join(depth_dir, traj)
                    
                    if not os.path.isdir(color_traj) or not os.path.exists(depth_traj):
                        continue
                        
                    # Find all RGB files
                    rgb_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
                        rgb_files.extend(glob.glob(os.path.join(color_traj, ext)))
                    
                    # Match with depth files
                    for rgb_file in rgb_files:
                        file_id = os.path.splitext(os.path.basename(rgb_file))[0]
                        depth_file = os.path.join(depth_traj, f"{file_id}.png")
                        
                        if os.path.exists(depth_file):
                            self.rgb_files.append(rgb_file)
                            self.depth_files.append(depth_file)
        
        # Split into train/val
        indices = list(range(len(self.rgb_files)))
        random.seed(42)
        random.shuffle(indices)
        
        split_idx = int(val_split * len(indices))
        selected_indices = indices[split_idx:] if is_train else indices[:split_idx]
        
        self.rgb_files = [self.rgb_files[i] for i in selected_indices]
        self.depth_files = [self.depth_files[i] for i in selected_indices]
        
        print(f"{'Training' if is_train else 'Validation'} examples: {len(self.rgb_files)}")
        
        # Create mmap files
        self._create_mmap_files()
        
        # Transforms (for augmentation only)
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]) if is_train else None
    
    def _create_mmap_files(self):
        """Create memory-mapped files for the dataset"""
        # Create temporary directory for mmap files
        self.mmap_dir = tempfile.mkdtemp()
        print(f"Creating mmap files in {self.mmap_dir}...")
        
        # Create mmap arrays
        num_samples = len(self.rgb_files)
        
        # Use the first image to determine shape
        rgb_shape = (num_samples, 3, self.image_size, self.image_size)
        depth_shape = (num_samples, 1, self.image_size, self.image_size)
        
        # Create mmap files
        self.rgb_mmap_path = os.path.join(self.mmap_dir, 'rgb_data.mmap')
        self.depth_mmap_path = os.path.join(self.mmap_dir, 'depth_data.mmap')
        
        # Initialize mmap arrays
        self.rgb_mmap = np.memmap(self.rgb_mmap_path, dtype=np.float32, mode='w+', 
                                  shape=rgb_shape)
        self.depth_mmap = np.memmap(self.depth_mmap_path, dtype=np.float32, mode='w+',
                                    shape=depth_shape)
        
        # Define transforms for preprocessing
        rgb_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        depth_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Process each file and store in mmap
        import tqdm
        for i, (rgb_file, depth_file) in enumerate(tqdm.tqdm(
                zip(self.rgb_files, self.depth_files), total=num_samples, 
                desc="Processing images for mmap")):
            
            # Load and transform RGB
            rgb_img = Image.open(rgb_file).convert('RGB')
            rgb_tensor = rgb_transform(rgb_img)
            
            # Load and transform depth
            depth_img = Image.open(depth_file)
            depth_tensor = depth_transform(depth_img)
            
            # Normalize depth using min-max
            depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
            depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Ensure single channel
            if depth_tensor.shape[0] > 1:
                depth_tensor = depth_tensor[0].unsqueeze(0)
            
            # Store in mmap
            self.rgb_mmap[i] = rgb_tensor.numpy()
            self.depth_mmap[i] = depth_tensor.numpy()
            
            # Save normalization parameters (needed for reproducibility)
            if not hasattr(self, 'depth_norm_params'):
                self.depth_norm_params = []
            self.depth_norm_params.append((depth_min.item(), depth_max.item()))
        
        # Flush to disk
        self.rgb_mmap.flush()
        self.depth_mmap.flush()
        
        # Reopen in read mode
        self.rgb_mmap = np.memmap(self.rgb_mmap_path, dtype=np.float32, mode='r', 
                                  shape=rgb_shape)
        self.depth_mmap = np.memmap(self.depth_mmap_path, dtype=np.float32, mode='r',
                                    shape=depth_shape)
        print("Memory mapping complete!")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Get data from mmap
        rgb_tensor = torch.from_numpy(self.rgb_mmap[idx].copy())
        depth_tensor = torch.from_numpy(self.depth_mmap[idx].copy())
        
        # Apply augmentation if needed
        if self.augmentation:
            # Stack for joint augmentation
            stacked = torch.cat([rgb_tensor, depth_tensor], dim=0)
            # Apply transforms that work on both
            if random.random() > 0.5:
                stacked = torch.flip(stacked, [2])  # Horizontal flip
            
            # Split back
            rgb_tensor = stacked[:3]
            depth_tensor = stacked[3:]
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'rgb_path': self.rgb_files[idx],
            'depth_path': self.depth_files[idx]
        }
    
    def cleanup(self):
        """Clean up mmap files"""
        del self.rgb_mmap
        del self.depth_mmap
        import shutil
        try:
            shutil.rmtree(self.mmap_dir)
            print(f"Cleaned up mmap directory: {self.mmap_dir}")
        except Exception as e:
            print(f"Error cleaning mmap directory: {e}")
