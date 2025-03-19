import os
import torch
from collections import OrderedDict
from optimized_data_loader import RAMCachedDataset

class FixedRAMCachedDataset(RAMCachedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Immediately stop the prefetching thread
        self.prefetch_running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
            self.prefetch_thread = None
        
        # Clear cache for pickling
        self.cache = {}
    
    def __getstate__(self):
        """Custom getstate to handle pickling"""
        state = self.__dict__.copy()
        # Remove the unpicklable entries
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

def create_single_process_dataloader(dataset, batch_size, shuffle=True, pin_memory=True):
    """Create a dataloader that doesn't use multiprocessing"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # No multiprocessing
        pin_memory=pin_memory
    )
