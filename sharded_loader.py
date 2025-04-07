import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

class ShardedDataset(Dataset):
    """Dataset that handles sharding for multi-worker loading"""
    
    def __init__(self, base_dataset, num_shards, shard_idx):
        self.base_dataset = base_dataset
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        
        # Calculate indices for this shard
        all_indices = list(range(len(base_dataset)))
        self.indices = [i for i in all_indices if i % num_shards == shard_idx]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

class PersistentShardedDataLoader:
    """DataLoader with persistent workers and sharding"""
    
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, 
                 pin_memory=True, prefetch_factor=2, persistent_workers=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create sharded datasets for each worker
        self.sharded_datasets = []
        for i in range(num_workers):
            self.sharded_datasets.append(ShardedDataset(dataset, num_workers, i))
        
        # Create dataloaders for each shard
        self.dataloaders = []
        for sharded_dataset in self.sharded_datasets:
            self.dataloaders.append(
                DataLoader(
                    sharded_dataset,
                    batch_size=batch_size // num_workers,
                    shuffle=shuffle,
                    num_workers=1,  # One worker per shard
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers
                )
            )
        
        # Create iterators
        self.iterators = None
    
    def __iter__(self):
        # Create/reset iterators for each dataloader
        self.iterators = [iter(loader) for loader in self.dataloaders]
        return self
    
    def __next__(self):
        # Try to get batch from each iterator
        batches = []
        for it in self.iterators:
            try:
                batches.append(next(it))
            except StopIteration:
                # If any iterator is exhausted, we're done
                if not batches:
                    raise StopIteration
        
        # Combine batches from all shards
        if not batches:
            raise StopIteration
            
        combined_batch = {}
        for key in batches[0].keys():
            if isinstance(batches[0][key], torch.Tensor):
                combined_batch[key] = torch.cat([batch[key] for batch in batches])
            elif isinstance(batches[0][key], list):
                combined_batch[key] = []
                for batch in batches:
                    combined_batch[key].extend(batch[key])
            else:
                # Handle other types as needed
                combined_batch[key] = [batch[key] for batch in batches]
        
        return combined_batch
    
    def __len__(self):
        # Length is the minimum length of all loaders
        return min(len(loader) for loader in self.dataloaders)
