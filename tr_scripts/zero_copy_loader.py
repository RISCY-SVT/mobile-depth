import torch
from torch.utils.data import DataLoader

class ZeroCopyDataLoader:
    """DataLoader that optimizes CPU-to-GPU transfers using zero-copy"""
    
    def __init__(self, dataset, batch_size, shuffle=True, pin_memory=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create underlying dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Allocate pinned memory for batches
        self.rgb_buffer = torch.empty(
            (batch_size, 3, dataset.image_size, dataset.image_size),
            dtype=torch.float32,
            pin_memory=True
        )
        
        self.depth_buffer = torch.empty(
            (batch_size, 1, dataset.image_size, dataset.image_size),
            dtype=torch.float32,
            pin_memory=True
        )
        
        # CUDA stream for async operations
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        # Get iterator from underlying dataloader
        self.loader_iter = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.loader_iter)
            
            with torch.cuda.stream(self.stream):
                # Extract tensors
                rgb = batch['rgb']
                depth = batch['depth']
                
                # Copy to pinned buffers
                self.rgb_buffer[:rgb.size(0)].copy_(rgb, non_blocking=True)
                self.depth_buffer[:depth.size(0)].copy_(depth, non_blocking=True)
                
                # Create CUDA tensors that reference the pinned memory
                rgb_cuda = self.rgb_buffer[:rgb.size(0)].cuda(non_blocking=True)
                depth_cuda = self.depth_buffer[:depth.size(0)].cuda(non_blocking=True)
                
                # Wait for transfer to complete
                self.stream.synchronize()
                
                # Replace tensors in batch with CUDA tensors
                batch['rgb'] = rgb_cuda
                batch['depth'] = depth_cuda
                
                return batch
                
        except StopIteration:
            raise StopIteration
    
    def __len__(self):
        return len(self.dataloader)
