import os, time, datetime, json, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from threading import Thread
import torch.nn.utils.prune as prune
import logging
import gc
import psutil

from improved_depth_loss import MultiScaleLoss, DepthWithSmoothnessLoss, RobustLoss, BerHuLoss
from improved_fix_data_loader import create_improved_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUMetrics:
    """Class to track GPU metrics in a background thread with improved reliability"""
    def __init__(self, device, interval=2.0):
        self.device = device
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics = {
            'timestamp': [],
            'memory_allocated': [],
            'memory_reserved': [],
            'utilization': []
        }
        # Защита от ошибок мониторинга
        self.last_error_time = 0
        self.error_count = 0
        
        # Внутренние методрики использования памяти
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0
    
    def start(self):
        if self.running:
            return
            
        # Reset tracking metrics
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0
        self.metrics = {
            'timestamp': [],
            'memory_allocated': [],
            'memory_reserved': [],
            'utilization': []
        }
            
        self.running = True
        self.thread = Thread(target=self._monitor_gpu)
        self.thread.daemon = True
        self.thread.start()
        logger.info("GPU monitoring started")
    
    def stop(self):
        if not self.running:
            return
            
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        logger.info(f"GPU monitoring stopped. Max allocated: {self.max_memory_allocated:.1f}MB, "
                   f"Max reserved: {self.max_memory_reserved:.1f}MB")
        
        # Saving final metrics to ensure we have the peak
        self._record_current_metrics()
    
    def _record_current_metrics(self):
        """Record current GPU metrics safely"""
        try:
            if self.device.type == 'cuda':
                current_time = time.time()
                
                # Memory metrics
                mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
                mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)  # MB
                
                # Update peak memory usage
                self.max_memory_allocated = max(self.max_memory_allocated, mem_allocated)
                self.max_memory_reserved = max(self.max_memory_reserved, mem_reserved)
                
                # Add to metrics history
                self.metrics['timestamp'].append(current_time)
                self.metrics['memory_allocated'].append(mem_allocated)
                self.metrics['memory_reserved'].append(mem_reserved)
                
                # GPU utilization via nvidia-smi (try-except for compatibility)
                try:
                    import subprocess
                    output = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        encoding='utf-8'
                    )
                    self.metrics['utilization'].append(float(output.strip()))
                except Exception:
                    # Fallback for systems without nvidia-smi
                    self.metrics['utilization'].append(0)
            
            # Reset error count on successful monitoring
            self.error_count = 0
                
        except Exception as e:
            # Error handling with exponential backoff
            current_time = time.time()
            if current_time - self.last_error_time > 10:
                # Reset error count after 10 seconds with no errors
                self.error_count = 0
            
            self.error_count += 1
            self.last_error_time = current_time
            
            # Log only occasionally to avoid flooding
            if self.error_count <= 3:
                logger.warning(f"GPU monitoring error: {e}")
    
    def _monitor_gpu(self):
        """Background thread for GPU monitoring with error resilience"""
        logger.info(f"GPU monitoring thread starting on device {self.device}")
        
        try:
            while self.running:
                # Record metrics
                self._record_current_metrics()
                
                # Add adaptive sleeping based on error frequency
                sleep_time = self.interval
                if self.error_count > 5:
                    # Increase interval if too many errors
                    sleep_time = min(30, self.interval * (1 + self.error_count // 5))
                
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"GPU monitoring thread crashed: {e}")
        finally:
            logger.info("GPU monitoring thread stopped")
    
    def get_current_usage(self):
        """Returns current GPU memory usage"""
        if not self.metrics['memory_allocated']:
            return 0, 0
        
        return (self.metrics['memory_allocated'][-1], 
                self.metrics['memory_reserved'][-1])
    
    def get_max_usage(self):
        """Returns maximum GPU memory usage seen so far"""
        return self.max_memory_allocated, self.max_memory_reserved
    
    def plot_metrics(self, output_dir):
        """Create and save GPU usage plots"""
        if not self.metrics['timestamp']:
            logger.warning("No GPU metrics collected - skipping plots")
            return
            
        # Create plot directory
        plot_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # Normalize timestamps
            start_time = self.metrics['timestamp'][0]
            time_mins = [(t - start_time) / 60.0 for t in self.metrics['timestamp']]  # minutes
            
            # Apply smoothing filter if we have enough data points
            if len(time_mins) > 10:
                kernel_size = min(10, len(time_mins) // 10)
                kernel = np.ones(kernel_size) / kernel_size
                mem_allocated_smooth = np.convolve(self.metrics['memory_allocated'], kernel, mode='valid')
                mem_reserved_smooth = np.convolve(self.metrics['memory_reserved'], kernel, mode='valid')
                time_mins_smooth = time_mins[kernel_size-1:]
            else:
                # Use raw data if not enough for smoothing
                mem_allocated_smooth = self.metrics['memory_allocated']
                mem_reserved_smooth = self.metrics['memory_reserved']
                time_mins_smooth = time_mins
            
            # Create memory plot with improved styling
            plt.figure(figsize=(12, 6))
            plt.plot(time_mins_smooth, mem_allocated_smooth, label='Allocated Memory (MB)', linewidth=2)
            plt.plot(time_mins_smooth, mem_reserved_smooth, label='Reserved Memory (MB)', linewidth=2)
            
            # Add peak memory indicator
            plt.axhline(y=self.max_memory_allocated, color='r', linestyle='--', 
                       label=f'Peak Allocated: {self.max_memory_allocated:.1f} MB')
            
            # Add GTX 1080 memory limit indicator (8GB)
            plt.axhline(y=8*1024, color='k', linestyle='-', alpha=0.3, 
                       label='GTX 1080 Limit (8GB)')
            
            # Add annotations
            plt.xlabel('Time (minutes)')
            plt.ylabel('Memory (MB)')
            plt.title('GPU Memory Usage During Training')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Add memory usage percentage on second y-axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.set_ylabel('Memory Usage (%)')
            ax2.set_ylim(0, 100)
            
            # Calculate percentage relative to 8GB
            mem_pct = np.array(mem_reserved_smooth) / (8*1024) * 100
            ax2.plot(time_mins_smooth, mem_pct, 'g--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'gpu_memory.png'), dpi=150)
            plt.close('all')  # Explicitly close to prevent memory leaks
            
            # Create utilization plot if available
            if any(self.metrics['utilization']):
                # Apply smoothing to utilization data
                if len(time_mins) > 10:
                    util_smooth = np.convolve(self.metrics['utilization'], kernel, mode='valid')
                else:
                    util_smooth = self.metrics['utilization']
                
                plt.figure(figsize=(12, 6))
                plt.plot(time_mins_smooth, util_smooth, label='GPU Utilization (%)', linewidth=2)
                plt.xlabel('Time (minutes)')
                plt.ylabel('Utilization (%)')
                plt.title('GPU Utilization During Training')
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                
                # Add horizontal lines at 25%, 50%, 75% for reference
                plt.axhline(y=25, color='r', linestyle='--', alpha=0.3)
                plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
                plt.axhline(y=75, color='r', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'gpu_utilization.png'), dpi=150)
                plt.close('all')
                
            logger.info(f"GPU metrics plots saved to {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error creating GPU metrics plots: {e}")
            plt.close('all')  # Ensure plots are closed even on error

def optimize_cuda_settings():
    """Apply optimized CUDA settings for stable training on GTX 1080"""
    # Set more conservative CUDA settings for stability on GTX 1080
    torch.backends.cudnn.benchmark = False  # Disable for more deterministic behavior
    torch.backends.cudnn.deterministic = True  # Enable for reproducibility
    
    # Enable TF32 precision only on newer GPUs (Ampere+)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # Disable TF32 on older GPUs like GTX 1080
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    # Improve memory allocator efficiency
    if torch.cuda.is_available():
        # More aggressive garbage collection for CUDA tensors
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy for PyTorch 1.10+
        if hasattr(torch.cuda, 'memory_stats'):
            # Use caching allocator only when we have 8GB+ GPU 
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_mem < 8.5:  # GTX 1080 has 8GB
                # Limiting max split size to avoid fragmentation on GTX 1080
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Check if we're running on Windows
    if os.name == 'nt':
        # On Windows, try to set process priority to high
        try:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            logger.info("Process priority set to HIGH")
        except Exception as e:
            logger.warning(f"Failed to set process priority: {e}")

def dump_cuda_state(tag: str = "") -> None:
    """
    Сохраняет расширенную информацию о состоянии CUDA памяти
    для диагностики проблем OOM.
    """
    if not torch.cuda.is_available():
        return
        
    # Create diagnostic directory if needed
    diag_dir = "cuda_diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = tag.replace(" ", "_")
    fn_base = os.path.join(diag_dir, f"cuda_mem_{ts}_{tag}" if tag else f"cuda_mem_{ts}")

    # Explicitly synchronize CUDA before getting memory stats
    torch.cuda.synchronize()
    
    # Basic summary
    with open(f"{fn_base}_summary.txt", "w") as f:
        f.write(torch.cuda.memory_summary())

    # Detailed statistics if available
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        with open(f"{fn_base}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    # Full memory snapshot if available
    if hasattr(torch.cuda, 'memory_snapshot'):
        try:
            snap = torch.cuda.memory_snapshot()
            with open(f"{fn_base}_snapshot.json", "w") as f:
                json.dump(snap, f, indent=2)
        except Exception as e:
            logger.error(f"Error getting memory snapshot: {e}")

    # Add system memory info
    try:
        with open(f"{fn_base}_system.txt", "w") as f:
            f.write(f"System memory: {psutil.virtual_memory()}\n")
            f.write(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB\n")
            f.write(f"CPU usage: {psutil.cpu_percent(interval=0.1)}%\n")
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")

    logger.warning(f"[OOM-DEBUG] CUDA state saved to {fn_base}_*.txt/json")

def cosine_learning_rate_with_warmup(optimizer, epoch, warmup_epochs, max_epochs, 
                                    init_lr, min_lr, warmup_start_lr=1e-6):
    """
    Cosine learning rate scheduler with warmup for more stable training, 
    especially important for quantization robustness.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = warmup_start_lr + (init_lr - warmup_start_lr) * epoch / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (max(1, max_epochs - warmup_epochs))
        lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    # Update learning rates
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def apply_weight_pruning(model, pruning_amount=0.3):
    """
    Apply temporary weight pruning to model to simulate quantization effects
    during training (can help with quantization-robustness).
    """
    pruned_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                pruned_layers += 1
            except Exception as e:
                logger.warning(f"Failed to prune layer {name}: {e}")
    
    logger.info(f"Applied temporary L1 pruning with amount {pruning_amount} to {pruned_layers} layers")

def remove_weight_pruning(model):
    """Remove pruning masks and make weights permanent"""
    removed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                try:
                    prune.remove(module, 'weight')
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove pruning from layer {name}: {e}")
    
    logger.info(f"Removed pruning masks from {removed_count} layers, pruned weights are now permanent")

def prepare_qat(model, num_bits=8):
    """
    Prepare model for Quantization-Aware Training (QAT).
    Важно делать это на поздних этапах обучения.
    """
    try:
        # Check if QAT is supported
        if not hasattr(torch.quantization, 'prepare_qat'):
            logger.warning("PyTorch version doesn't support prepare_qat. Skipping QAT preparation.")
            return model
        
        # Configure quantization settings
        if num_bits == 8:
            # Standard INT8 quantization
            qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        elif num_bits == 16:
            # FP16 quantization (less aggressive)
            qconfig = torch.quantization.float_qparams_weight_only_qconfig
        else:
            # Fallback to standard INT8
            logger.warning(f"Unsupported num_bits={num_bits}, using INT8 instead")
            qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        
        # Apply QAT to model
        model.qconfig = qconfig
        torch.quantization.prepare_qat(model, inplace=True)
        
        logger.info(f"Model prepared for QAT with {num_bits} bits")
        
        # Force a garbage collection after QAT preparation
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Failed to prepare model for QAT: {e}")
        
    return model

def train_epoch_improved(model, train_loader, criterion, optimizer, scaler, device, 
                         epoch, max_epochs, mixed_precision=False, max_grad_norm=None, 
                         update_lr_fn=None, batch_visualize_freq=500, gpu_metrics=None,
                         memory_efficient=True):
    """Улучшенная функция обучения эпохи с оптимизацией памяти для GTX 1080"""
    # Reset statistics and clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    # Set model to training mode
    model.train()
    
    # Initialize counters and timers
    epoch_loss = 0
    epoch_start = time.time()
    total_vis_loss = 0
    vis_batches = 0
    
    # Prefetcher setup for efficient data loading
    prefetcher = QATPrefetcher(train_loader, device) if 'QATPrefetcher' in globals() else None
    
    # Set up progress bar
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epochs}")
    
    # Performance tracking
    batch_times = []
    forward_times = []
    backward_times = []
    data_times = []
    last_time = time.time()
    
    # Update learning rate if needed
    if update_lr_fn is not None:
        current_lr = update_lr_fn(optimizer, epoch)
        logger.info(f"Set learning rate to {current_lr:.6f}")
    
    # Batch processing loop
    batch_idx = 0
    
    # Get initial batch
    if prefetcher:
        batch = prefetcher.next()
    else:
        # Use standard iteration if no prefetcher
        dataloader_iter = iter(train_loader)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            batch = None
    
    # Memory usage warning threshold (80% of GPU memory)
    memory_warning_threshold = 0.8 * torch.cuda.get_device_properties(0).total_memory / (1024**2)
    memory_alert_shown = False
    
    while batch is not None:
        try:
            # Record data loading time
            data_time = time.time() - last_time
            data_times.append(data_time)
            
            # Memory-efficient mode processes smaller batches if needed
            if memory_efficient and gpu_metrics:
                current_mem, _ = gpu_metrics.get_current_usage()
                
                # Show warning if approaching memory limit
                if current_mem > memory_warning_threshold and not memory_alert_shown:
                    logger.warning(f"GPU memory usage high: {current_mem:.1f}MB / {memory_warning_threshold:.1f}MB")
                    memory_alert_shown = True
            
            # Transfer data to device with non-blocking for better parallelism
            rgb = batch['rgb'].to(device, non_blocking=True)
            depth_gt = batch['depth'].to(device, non_blocking=True)
            
            # Make sure transfer is complete
            torch.cuda.synchronize()
            
            # Reset gradients - more efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with error handling
            forward_start = time.time()
            
            try:
                if mixed_precision:
                    with torch.amp.autocast('cuda'):
                        depth_pred = model(rgb)
                        loss = criterion(depth_pred, depth_gt, rgb)
                else:
                    depth_pred = model(rgb)
                    loss = criterion(depth_pred, depth_gt, rgb)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"WARNING: Loss is NaN or Inf at batch {batch_idx}. Skipping.")
                    batch_idx += 1
                    
                    # Get next batch
                    if prefetcher:
                        batch = prefetcher.next()
                    else:
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            batch = None
                            
                    continue
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at epoch {epoch} batch {batch_idx}")
                    torch.cuda.synchronize()
                
                # Diagnose the OOM issue
                dump_cuda_state(tag=f"ep{epoch}_b{batch_idx}_fwd")
                logger.error(f"Error during forward pass at batch {batch_idx}: {e}")
                
                # Release memory and try to continue with next batch
                del rgb, depth_gt
                if 'depth_pred' in locals():
                    del depth_pred
                torch.cuda.empty_cache()
                
                batch_idx += 1
                if prefetcher:
                    batch = prefetcher.next()
                else:
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        batch = None
                        
                continue
                
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Backward pass with error handling
            backward_start = time.time()
            
            try:
                if mixed_precision:
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    # Extra synchronization for stability
                    torch.cuda.synchronize()
                    
                    if max_grad_norm is not None:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Handle NaN/Inf gradients
                        for param in model.parameters():
                            if param.grad is not None:
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Clip gradients for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Ensure processing is complete
                    torch.cuda.synchronize()
                    
                    # Optimizer step with error handling
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(f"OOM during optimizer step at epoch {epoch} batch {batch_idx}")
                            torch.cuda.synchronize()
                        
                        dump_cuda_state(tag=f"ep{epoch}_b{batch_idx}_opt")
                        logger.error(f"Error during optimizer step at batch {batch_idx}: {e}")
                        
                        # Reset scaler if there was an issue
                        scaler._scale = torch.tensor(1.0, device=device)
                        torch.cuda.empty_cache()
                        
                        batch_idx += 1
                        if prefetcher:
                            batch = prefetcher.next()
                        else:
                            try:
                                batch = next(dataloader_iter)
                            except StopIteration:
                                batch = None
                                
                        continue
                        
                else:
                    # Standard backward pass
                    loss.backward()
                    
                    if max_grad_norm is not None:
                        # Handle NaN/Inf gradients
                        for param in model.parameters():
                            if param.grad is not None:
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Clip gradients for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Standard optimizer step
                    optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM during backward pass at epoch {epoch} batch {batch_idx}")
                    torch.cuda.synchronize()
                
                dump_cuda_state(tag=f"ep{epoch}_b{batch_idx}_bwd")
                logger.error(f"Error during backward pass at batch {batch_idx}: {e}")
                
                # Clean up references to free memory
                del rgb, depth_gt, depth_pred, loss
                torch.cuda.empty_cache()
                
                batch_idx += 1
                if prefetcher:
                    batch = prefetcher.next()
                else:
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        batch = None
                        
                continue
                
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Calculate batch time
            batch_time = time.time() - last_time
            batch_times.append(batch_time)
            last_time = time.time()
            
            # Update statistics
            loss_value = loss.item()
            epoch_loss += loss_value
            
            # Update visualization loss statistics
            total_vis_loss += loss_value
            vis_batches += 1
            
            # Generate batch visualization
            if batch_idx % batch_visualize_freq == 0 and hasattr(criterion, '_visualize_batch'):
                try:
                    criterion._visualize_batch(depth_pred, depth_gt, rgb, batch_idx, epoch)
                except Exception as e:
                    logger.warning(f"Error during batch visualization: {e}")
            
            # Clean up to prevent memory fragmentation
            del rgb, depth_gt, depth_pred, loss
            
            # Update progress bar with rolling averages
            num_samples = min(len(batch_times), 20)  # Average over last 20 batches
            avg_forward = sum(forward_times[-num_samples:]) / num_samples
            avg_backward = sum(backward_times[-num_samples:]) / num_samples
            avg_data = sum(data_times[-num_samples:]) / num_samples
            avg_batch = sum(batch_times[-num_samples:]) / num_samples
            avg_loss = total_vis_loss / vis_batches
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'batch': f"{avg_batch:.3f}s",
                'fw': f"{avg_forward:.3f}s",
                'bw': f"{avg_backward:.3f}s",
                'data': f"{avg_data:.3f}s"
            })
                
            progress_bar.update(1)
            
            # Periodic logging
            if batch_idx % 20 == 0:
                # Get current GPU memory usage
                mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                mem_max = torch.cuda.max_memory_allocated(device) / (1024**2)
                
                logger.info(f"Epoch {epoch+1}/{max_epochs} - Batch {batch_idx} - "
                           f"Loss: {avg_loss:.4f}, GPU Memory: {mem_allocated:.1f}MB / {mem_max:.1f}MB")
                
            # Periodic GPU memory cleanup to reduce fragmentation
            if batch_idx % 30 == 0:
                torch.cuda.empty_cache()
                # Force garbage collection
                gc.collect()
            
            # Get next batch
            batch_idx += 1
            if prefetcher:
                batch = prefetcher.next()
            else:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    batch = None
                
        except Exception as e:
            # Handle generic exceptions
            if "out of memory" in str(e):
                logger.error(f"OOM at epoch {epoch} batch {batch_idx}")
                torch.cuda.synchronize()
            
            dump_cuda_state(tag=f"ep{epoch}_b{batch_idx}_err")
            logger.error(f"Unexpected error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up and try to continue
            torch.cuda.empty_cache()
            
            batch_idx += 1
            if prefetcher:
                batch = prefetcher.next()
            else:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    batch = None
                    
            continue
    
    # Close progress bar
    progress_bar.close()
    
    # Calculate epoch statistics
    num_processed_batches = max(batch_idx, 1)
    avg_loss = epoch_loss / num_processed_batches
    
    # Log performance statistics if available
    if len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_backward_time = sum(backward_times) / len(backward_times)
        avg_data_time = sum(data_times) / len(data_times)
        
        logger.info(f"Average times - Batch: {avg_batch_time:.3f}s, "
              f"Forward: {avg_forward_time:.3f}s, "
              f"Backward: {avg_backward_time:.3f}s, "
              f"Data: {avg_data_time:.3f}s")
    
    # Final cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    return avg_loss

def validate_improved(model, val_loader, criterion, device, output_dir=None, epoch=None, 
                     gpu_metrics=None, memory_efficient=True):
    """Улучшенная функция валидации с оптимизацией памяти для GTX 1080"""
    # Reset CUDA state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    val_loss = 0
    abs_rel_error = 0
    rmse = 0
    
    # Use prefetcher if available
    prefetcher = QATPrefetcher(val_loader, device) if 'QATPrefetcher' in globals() else None
    
    # Setup progress bar
    progress_bar = tqdm(total=len(val_loader), desc="Validation")
    
    # Initialize batch handling
    batch_idx = 0
    visualized = False
    error_count = 0
    max_errors = 5  # Maximum tolerated errors before giving up
    
    # Get first batch
    if prefetcher:
        batch = prefetcher.next()
    else:
        # Use standard iteration if no prefetcher
        try:
            dataloader_iter = iter(val_loader)
            batch = next(dataloader_iter)
        except StopIteration:
            batch = None
            
    # Memory warning threshold (in MB)
    memory_warning_threshold = 0.8 * torch.cuda.get_device_properties(0).total_memory / (1024**2)
    
    with torch.no_grad():
        while batch is not None and error_count < max_errors:
            try:
                # Check memory usage in memory-efficient mode
                if memory_efficient and gpu_metrics:
                    current_mem, _ = gpu_metrics.get_current_usage()
                    if current_mem > memory_warning_threshold:
                        logger.warning(f"High GPU memory during validation: {current_mem:.1f}MB")
                        # Force cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Explicitly sync before validation step
                torch.cuda.synchronize()
                
                # Transfer data to device
                rgb = batch['rgb'].to(device, non_blocking=True)
                depth_gt = batch['depth'].to(device, non_blocking=True)
                
                # Forward pass with explicit error handling
                try:
                    depth_pred = model(rgb)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"OOM during validation forward pass at batch {batch_idx}")
                        torch.cuda.synchronize()
                        
                    dump_cuda_state(tag=f"val_ep{epoch}_b{batch_idx}")
                    logger.error(f"Error during validation forward pass: {e}")
                    
                    # Clean up and try to continue
                    del rgb, depth_gt
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    error_count += 1
                    batch_idx += 1
                    
                    # Get next batch
                    if prefetcher:
                        batch = prefetcher.next()
                    else:
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            batch = None
                            
                    continue
                
                # Sync after forward pass
                torch.cuda.synchronize()
                
                # Calculate loss with careful handling of multi-scale outputs
                try:
                    if isinstance(depth_pred, list):
                        depth_pred_solo = depth_pred[0]
                        loss = criterion(depth_pred, depth_gt, rgb)
                    else:
                        depth_pred_solo = depth_pred
                        loss = criterion(depth_pred, depth_gt, rgb)
                    
                    # Update metrics
                    val_loss += float(loss.item())
                except Exception as e:
                    logger.error(f"Error calculating validation loss: {e}")
                    error_count += 1
                    
                    # Try to continue with next batch
                    batch_idx += 1
                    if prefetcher:
                        batch = prefetcher.next()
                    else:
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            batch = None
                            
                    continue
                
                # Calculate metrics with explicit error handling
                try:
                    # Absolute relative error
                    rel_err = torch.abs(depth_pred_solo - depth_gt) / (depth_gt + 1e-6)
                    abs_rel_error += float(torch.mean(rel_err).item())
                    
                    # RMSE
                    rmse_batch = torch.sqrt(torch.mean((depth_pred_solo - depth_gt) ** 2))
                    rmse += float(rmse_batch.item())
                except Exception as e:
                    logger.warning(f"Error calculating validation metrics: {e}")
                    # Continue even if metrics calculation fails
                
                # Generate visualization
                if output_dir and not visualized and epoch is not None:
                    try:
                        save_visualization(rgb, depth_gt, depth_pred_solo, output_dir, epoch)
                        visualized = True
                    except Exception as e:
                        logger.warning(f"Error generating validation visualization: {e}")
                
                # Update progress bar
                progress_bar.update(1)
                
                # Clean up references to free memory
                del rgb, depth_gt, depth_pred, depth_pred_solo, loss
                
                # Get next batch
                batch_idx += 1
                if prefetcher:
                    batch = prefetcher.next()
                else:
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        batch = None
                
                # Periodic memory cleanup
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Unexpected error in validation batch {batch_idx}: {e}")
                dump_cuda_state(tag=f"val_err_ep{epoch}_b{batch_idx}")
                
                error_count += 1
                batch_idx += 1
                
                # Try to continue with next batch
                try:
                    if prefetcher:
                        batch = prefetcher.next()
                    else:
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            batch = None
                except:
                    batch = None
    
    # Close progress bar
    progress_bar.close()
    
    # Final cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Calculate average metrics
    num_processed_batches = max(batch_idx - error_count, 1)
    avg_loss = val_loss / num_processed_batches
    avg_abs_rel = abs_rel_error / num_processed_batches
    avg_rmse = rmse / num_processed_batches
    
    logger.info(f"Validation complete - Batches processed: {batch_idx}, Errors: {error_count}")
    
    return avg_loss, avg_abs_rel, avg_rmse

def save_visualization(rgb, depth_gt, depth_pred, output_dir, epoch):
    """Enhanced visualization with optimized memory usage"""
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Transfer data to CPU before visualization processing
    with torch.no_grad():
        rgb_np = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
        depth_gt_np = depth_gt[0, 0].detach().cpu().numpy()
        depth_pred_np = depth_pred[0, 0].detach().cpu().numpy()
    
    # Release GPU tensors explicitly
    del rgb, depth_gt, depth_pred
    torch.cuda.empty_cache()
    
    # Denormalize RGB image
    rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    
    # Calculate error map
    error_map = np.abs(depth_pred_np - depth_gt_np)
    
    # Calculate metrics for this visualization
    rel_error = np.mean(np.abs(depth_pred_np - depth_gt_np) / (depth_gt_np + 1e-6))
    rmse = np.sqrt(np.mean((depth_pred_np - depth_gt_np) ** 2))
    
    # Create visualization with matplotlib
    try:
        plt.figure(figsize=(15, 10))
        
        # RGB input
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_np)
        plt.title('RGB Input')
        plt.axis('off')
        
        # Ground truth depth
        plt.subplot(2, 2, 2)
        plt.imshow(depth_gt_np, cmap='plasma')
        plt.title(f'Ground Truth Depth (min={depth_gt_np.min():.3f}, max={depth_gt_np.max():.3f})')
        plt.axis('off')
        plt.colorbar()
        
        # Predicted depth
        plt.subplot(2, 2, 3)
        plt.imshow(depth_pred_np, cmap='plasma')
        plt.title(f'Predicted Depth (min={depth_pred_np.min():.3f}, max={depth_pred_np.max():.3f})')
        plt.axis('off')
        plt.colorbar()
        
        # Error map
        plt.subplot(2, 2, 4)
        plt.imshow(error_map, cmap='hot')
        plt.title(f'Absolute Error (RelErr={rel_error:.4f}, RMSE={rmse:.4f})')
        plt.axis('off')
        plt.colorbar()
        
        # Add epoch information
        plt.suptitle(f'Epoch {epoch+1} - B_MobileDepth Results', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png'), dpi=150)
        plt.close('all')  # Explicitly close to free memory
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
    finally:
        # Clean up memory
        plt.close('all')
        
    # Release numpy arrays
    del rgb_np, depth_gt_np, depth_pred_np, error_map
    gc.collect()

def quantization_test(model, test_loader, device, num_samples=5, output_dir=None):
    """
    Simulate quantization effects on the model with memory-efficient implementation
    for GTX 1080.
    """
    logger.info("Testing model with simulated quantization...")
    
    # Ensure model is in evaluation mode
    model = model.to(device).float().eval()
    
    # Create visualization directory
    if output_dir:
        quant_vis_dir = os.path.join(output_dir, 'quantization_test')
        os.makedirs(quant_vis_dir, exist_ok=True)
    
    try:
        # Create a copy of the model for quantization simulation
        import copy
        model_quant = copy.deepcopy(model)
        
        # Force model to CPU for quantization (smaller memory footprint)
        model_quant = model_quant.cpu()
        
        # Configure quantization
        model_quant.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model_quant, inplace=True)
        
        # Convert to quantized model (on CPU)
        torch.quantization.convert(model_quant, inplace=True)
        
        # Progress tracking
        logger.info(f"Testing {num_samples} samples with both original and quantized models")
        results = []
        
        # Use limited number of samples
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Get data
            rgb = batch['rgb'].to(device, non_blocking=True)
            depth_gt = batch['depth'].to(device, non_blocking=True)
            
            # Original model inference
            with torch.no_grad():
                # Ensure GPU memory is sufficient
                torch.cuda.empty_cache()
                
                try:
                    depth_pred_original = model(rgb)
                    if isinstance(depth_pred_original, list):
                        depth_pred_original = depth_pred_original[0]
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error("OOM during original model inference")
                        dump_cuda_state(tag=f"quant_test_orig_{i}")
                        
                        # Skip this sample
                        continue
            
            # Quantized model inference on CPU
            with torch.no_grad():
                # Move data to CPU for quantized model
                rgb_cpu = rgb.cpu()
                
                try:
                    depth_pred_quant = model_quant(rgb_cpu)
                    if isinstance(depth_pred_quant, list):
                        depth_pred_quant = depth_pred_quant[0]
                    
                    # Move quantized output to GPU for comparison
                    depth_pred_quant = depth_pred_quant.to(device)
                except Exception as e:
                    logger.error(f"Error during quantized model inference: {e}")
                    continue
            
            # Calculate metrics
            try:
                rel_err_original = torch.abs(depth_pred_original - depth_gt) / (depth_gt + 1e-6)
                rel_err_quant = torch.abs(depth_pred_quant - depth_gt) / (depth_gt + 1e-6)
                
                mean_rel_original = torch.mean(rel_err_original).item()
                mean_rel_quant = torch.mean(rel_err_quant).item()
                
                rmse_original = torch.sqrt(torch.mean((depth_pred_original - depth_gt) ** 2)).item()
                rmse_quant = torch.sqrt(torch.mean((depth_pred_quant - depth_gt) ** 2)).item()
                
                results.append({
                    'sample': i+1,
                    'rel_err_original': mean_rel_original,
                    'rel_err_quant': mean_rel_quant,
                    'rmse_original': rmse_original,
                    'rmse_quant': rmse_quant,
                    'degradation_rel': (mean_rel_quant - mean_rel_original) / mean_rel_original * 100,
                    'degradation_rmse': (rmse_quant - rmse_original) / rmse_original * 100
                })
                
                logger.info(f"Sample {i+1} - Original: RelErr={mean_rel_original:.4f}, RMSE={rmse_original:.4f}, "
                          f"Quantized: RelErr={mean_rel_quant:.4f}, RMSE={rmse_quant:.4f}")
            
            except Exception as e:
                logger.error(f"Error calculating metrics for sample {i+1}: {e}")
                continue
            
            # Save visualization if requested
            if output_dir:
                try:
                    # Move tensors to CPU for visualization
                    rgb_np = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
                    depth_gt_np = depth_gt[0, 0].detach().cpu().numpy()
                    depth_pred_original_np = depth_pred_original[0, 0].detach().cpu().numpy()
                    depth_pred_quant_np = depth_pred_quant[0, 0].detach().cpu().numpy()
                    
                    # Free GPU memory
                    del rgb, depth_gt, depth_pred_original, depth_pred_quant
                    torch.cuda.empty_cache()
                    
                    # Denormalize RGB
                    rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                    
                    # Calculate error maps
                    error_original = np.abs(depth_pred_original_np - depth_gt_np)
                    error_quant = np.abs(depth_pred_quant_np - depth_gt_np)
                    
                    # Create visualization
                    plt.figure(figsize=(15, 10))
                    
                    # Input image
                    plt.subplot(3, 2, 1)
                    plt.imshow(rgb_np)
                    plt.title('RGB Input')
                    plt.axis('off')
                    
                    # Ground truth
                    plt.subplot(3, 2, 2)
                    plt.imshow(depth_gt_np, cmap='plasma')
                    plt.title('Ground Truth Depth')
                    plt.axis('off')
                    
                    # Original prediction
                    plt.subplot(3, 2, 3)
                    plt.imshow(depth_pred_original_np, cmap='plasma')
                    plt.title(f'Original: RelErr={mean_rel_original:.4f}, RMSE={rmse_original:.4f}')
                    plt.axis('off')
                    
                    # Quantized prediction
                    plt.subplot(3, 2, 4)
                    plt.imshow(depth_pred_quant_np, cmap='plasma')
                    plt.title(f'Quantized: RelErr={mean_rel_quant:.4f}, RMSE={rmse_quant:.4f}')
                    plt.axis('off')
                    
                    # Original error
                    plt.subplot(3, 2, 5)
                    plt.imshow(error_original, cmap='hot')
                    plt.title('Original Error Map')
                    plt.axis('off')
                    
                    # Quantized error
                    plt.subplot(3, 2, 6)
                    plt.imshow(error_quant, cmap='hot')
                    plt.title('Quantized Error Map')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(quant_vis_dir, f'quantization_test_{i+1}.png'), dpi=150)
                    plt.close('all')
                    
                    # Clean up memory
                    del rgb_np, depth_gt_np, depth_pred_original_np, depth_pred_quant_np
                    del error_original, error_quant
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error creating quantization visualization: {e}")
        
        # Generate summary report
        if results:
            try:
                avg_rel_original = sum(r['rel_err_original'] for r in results) / len(results)
                avg_rel_quant = sum(r['rel_err_quant'] for r in results) / len(results)
                avg_rmse_original = sum(r['rmse_original'] for r in results) / len(results)
                avg_rmse_quant = sum(r['rmse_quant'] for r in results) / len(results)
                
                logger.info("\nQuantization Test Summary:")
                logger.info(f"Samples tested: {len(results)}")
                logger.info(f"Original model: Avg RelErr={avg_rel_original:.4f}, Avg RMSE={avg_rmse_original:.4f}")
                logger.info(f"Quantized model: Avg RelErr={avg_rel_quant:.4f}, Avg RMSE={avg_rmse_quant:.4f}")
                logger.info(f"Average RelErr degradation: {(avg_rel_quant-avg_rel_original)/avg_rel_original*100:.2f}%")
                logger.info(f"Average RMSE degradation: {(avg_rmse_quant-avg_rmse_original)/avg_rmse_original*100:.2f}%")
                
                # Save summary to file
                if output_dir:
                    with open(os.path.join(quant_vis_dir, 'quantization_summary.txt'), 'w') as f:
                        f.write("Quantization Test Summary\n")
                        f.write("==========================\n\n")
                        f.write(f"Samples tested: {len(results)}\n")
                        f.write(f"Original model: Avg RelErr={avg_rel_original:.4f}, Avg RMSE={avg_rmse_original:.4f}\n")
                        f.write(f"Quantized model: Avg RelErr={avg_rel_quant:.4f}, Avg RMSE={avg_rmse_quant:.4f}\n")
                        f.write(f"Average RelErr degradation: {(avg_rel_quant-avg_rel_original)/avg_rel_original*100:.2f}%\n")
                        f.write(f"Average RMSE degradation: {(avg_rmse_quant-avg_rmse_original)/avg_rmse_original*100:.2f}%\n\n")
                        
                        f.write("Sample-by-sample results:\n")
                        for r in results:
                            f.write(f"Sample {r['sample']}: Original (RelErr={r['rel_err_original']:.4f}, RMSE={r['rmse_original']:.4f}), "
                                   f"Quantized (RelErr={r['rel_err_quant']:.4f}, RMSE={r['rmse_quant']:.4f}), "
                                   f"Degradation: RelErr={r['degradation_rel']:.2f}%, RMSE={r['degradation_rmse']:.2f}%\n")
            except Exception as e:
                logger.error(f"Error generating quantization summary: {e}")
        
        # Final cleanup
        del model_quant
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("Quantization test completed.")
        return True
    
    except Exception as e:
        logger.error(f"Error during quantization test: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        
        return False

def train_improved(args, model, train_dataset, val_dataset):
    """
    Улучшенная функция тренировки с оптимизацией памяти,
    адаптированная специально для GTX 1080 и B_MobileDepth модели.
    """
    # Setup output directory structure
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Configure logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, 'logs', 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log memory information at the beginning of the training
    total_system_memory_gb = psutil.virtual_memory().total / (1024**3)
    free_system_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    logger.info(f"System memory: Total: {total_system_memory_gb:.1f}GB, "
              f"Available: {free_system_memory_gb:.1f}GB")
    
    # Apply CUDA optimizations
    optimize_cuda_settings()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Log GPU memory details
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU memory: {gpu_mem_gb:.2f} GB")
        
        # Check if this is likely a GTX 1080
        if 7.5 <= gpu_mem_gb <= 8.5:
            logger.info("Detected GTX 1080 or similar 8GB GPU - using memory-efficient mode")
            memory_efficient = True
        else:
            memory_efficient = False
    else:
        # No CUDA available
        memory_efficient = False
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function
    if hasattr(model, 'training') and model.training and hasattr(args, 'use_robust_loss') and args.use_robust_loss:
        # Use MultiScaleLoss with RobustLoss
        criterion = MultiScaleLoss(
            base_criterion=RobustLoss(
                alpha=0.4,     # L1
                beta=0.3,      # BerHu
                gamma=0.2,     # Scale-Invariant
                delta=0.1,     # SSIM
                edge_weight=0.15  # Edge-Aware
            )
        )
        logger.info("Using MultiScaleLoss with RobustLoss")
    else:
        # Use MultiScaleLoss with DepthWithSmoothnessLoss
        criterion = MultiScaleLoss(
            base_criterion=DepthWithSmoothnessLoss(
                base_weight=0.7,
                smoothness_weight=0.15,
                gradient_weight=0.15
            ),
            weights=[0.6, 0.2, 0.1, 0.1]  # Weights for different scales
        )
        logger.info("Using MultiScaleLoss with DepthWithSmoothnessLoss")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-3  # Higher epsilon for better numerical stability
    )
    
    # Set up gradient scaler for mixed precision
    scaler = torch.amp.GradScaler() if args.mixed_precision else None
    
    # Set up data loaders
    if 'create_improved_dataloader' in globals():
        # Use improved dataloader if available
        train_loader = create_improved_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
            prefetch_factor=args.prefetch_factor if hasattr(args, 'prefetch_factor') else 2
        )
        
        val_loader = create_improved_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=max(1, args.num_workers // 2) if hasattr(args, 'num_workers') else 2
        )
    else:
        # Use standard PyTorch DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
            num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
            num_workers=max(1, args.num_workers // 2) if hasattr(args, 'num_workers') else 2
        )
    
    # Initialize GPU metrics tracking
    gpu_metrics = GPUMetrics(device)
    gpu_metrics.start()
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
        try:
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if this is a valid checkpoint
            if all(k in checkpoint for k in ['epoch', 'model_state_dict', 'optimizer_state_dict']):
                start_epoch = checkpoint['epoch'] + 1
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                
                # Load model and optimizer states
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                logger.info(f"Resumed from epoch {start_epoch-1}")
                
                # If using mixed precision, try to load scaler state
                if args.mixed_precision and scaler and 'scaler_state_dict' in checkpoint:
                    try:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        logger.info("Resumed gradient scaler state")
                    except Exception as e:
                        logger.warning(f"Could not load scaler state: {e}")
            else:
                logger.warning("Invalid checkpoint format - starting from scratch")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    # Set up learning rate scheduler parameters
    num_warmup_epochs = min(5, args.epochs // 10)  # Scale warmup with total epochs
    
    # Learning rate scheduler update function
    def update_lr(optimizer, epoch):
        return cosine_learning_rate_with_warmup(
            optimizer, epoch, 
            warmup_epochs=num_warmup_epochs, 
            max_epochs=args.epochs, 
            init_lr=args.lr, 
            min_lr=args.min_lr
        )
    
    # Set up logging for training progress
    log_file = os.path.join(output_dir, 'logs', 'training_log.csv')
    with open(log_file, 'a') as f:
        if start_epoch == 0:  # Only write header if starting fresh
            f.write("epoch,time,lr,train_loss,val_loss,abs_rel,rmse,gpu_util,gpu_mem_used,gpu_mem_peak\n")
    
    logger.info("Starting training...")
    training_start_time = time.time()
    
    # Variables for QAT
    qat_started = False
    qat_start_epoch = int(args.epochs * 0.75)  # Start QAT at 75% of training
    pruning_percentage = 0.3  # Initial pruning percentage
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        # Perform memory cleanup before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get current learning rate
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}, LR: {current_lr:.6f}")
        torch.cuda.synchronize()
        
        # Start Quantization-Aware Training at later stages
        if epoch >= qat_start_epoch and not qat_started and hasattr(args, 'use_qat') and args.use_qat:
            logger.info(f"Starting Quantization-Aware Training at epoch {epoch+1}")
            try:
                model = prepare_qat(model)
                qat_started = True
            except Exception as e:
                logger.error(f"Failed to start QAT: {e}")
        
        # Apply temporary weight pruning to simulate quantization effects
        if epoch > 0 and epoch % 5 == 0 and pruning_percentage < 0.5:
            try:
                # Gradually increase pruning percentage to prepare the model
                pruning_percentage += 0.05
                apply_weight_pruning(model, pruning_percentage)
                
                # Train for one mini-batch with pruning (reduced batch size for stability)
                mini_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=max(1, args.batch_size // 2),
                    shuffle=True, 
                    num_workers=1
                )
                
                # Train for just a few batches with pruning
                mini_epoch = train_epoch_improved(
                    model, mini_train_loader, criterion, optimizer, scaler, device,
                    epoch, args.epochs, mixed_precision=args.mixed_precision,
                    max_grad_norm=args.clip_grad, update_lr_fn=None,
                    gpu_metrics=gpu_metrics, memory_efficient=memory_efficient
                )
                
                # Remove pruning masks but keep weights pruned
                remove_weight_pruning(model)
                
                # Log the effect of pruning
                logger.info(f"Applied {pruning_percentage:.2f} pruning with mini-training loss: {mini_epoch:.4f}")
                
                # Force cleanup after pruning session
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error during pruning step: {e}")
                # Continue training even if pruning fails
                torch.cuda.empty_cache()
                gc.collect()
            
        # Train one epoch
        try:
            train_loss = train_epoch_improved(
                model, train_loader, criterion, optimizer, scaler, device, 
                epoch, args.epochs, mixed_precision=args.mixed_precision,
                max_grad_norm=args.clip_grad, update_lr_fn=update_lr,
                gpu_metrics=gpu_metrics, memory_efficient=memory_efficient
            )
        except Exception as e:
            logger.error(f"Error during training epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save checkpoint before potential crash
            emergency_checkpoint_path = os.path.join(output_dir, 'checkpoints', f'emergency_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': float('inf'),  # Mark as incomplete
                'best_val_loss': best_val_loss
            }, emergency_checkpoint_path)
            logger.info(f"Saved emergency checkpoint at epoch {epoch+1}")
            
            # Clean up and continue to next epoch
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # Validate if needed
        val_loss, abs_rel, rmse = float('nan'), float('nan'), float('nan')
        if (epoch + 1) % args.val_freq == 0:
            try:
                val_loss, abs_rel, rmse = validate_improved(
                    model, val_loader, criterion, device, 
                    output_dir, epoch,
                    gpu_metrics=gpu_metrics, 
                    memory_efficient=memory_efficient
                )
                
                # Echo metrics
                epoch_time = time.time() - epoch_start_time
                
                # Get GPU metrics
                gpu_util = np.mean(gpu_metrics.metrics['utilization'][-100:]) if gpu_metrics.metrics['utilization'] else 0
                gpu_mem = np.mean(gpu_metrics.metrics['memory_allocated'][-100:]) if gpu_metrics.metrics['memory_allocated'] else 0
                gpu_mem_peak = gpu_metrics.max_memory_allocated
                
                # Log metrics
                with open(log_file, 'a') as f:
                    f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},"
                            f"{val_loss:.4f},{abs_rel:.4f},{rmse:.4f},{gpu_util:.1f},{gpu_mem:.1f},{gpu_mem_peak:.1f}\n")
                
                # Print summary
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
                logger.info(f"Training loss: {train_loss:.4f}")
                logger.info(f"Validation - Loss: {val_loss:.4f}, AbsRel: {abs_rel:.4f}, RMSE: {rmse:.4f}")
                logger.info(f"GPU memory - Current: {gpu_mem:.1f}MB, Peak: {gpu_mem_peak:.1f}MB")
                
                # Save best model
                if not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'val_loss': val_loss,
                        'abs_rel': abs_rel,
                        'rmse': rmse,
                        'best_val_loss': best_val_loss
                    }, best_model_path)
                    logger.info(f"Saved best model with validation loss {val_loss:.4f}")
            except Exception as e:
                logger.error(f"Error during validation: {e}")
                import traceback
                traceback.print_exc()
                
                # Just log training metrics when validation fails
                epoch_time = time.time() - epoch_start_time
                with open(log_file, 'a') as f:
                    f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},,,,,\n")
                
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
                logger.info(f"Training loss: {train_loss:.4f}")
                logger.info("Validation failed - continuing training")
                
                # Force cleanup after validation error
                torch.cuda.empty_cache()
                gc.collect()
        else:
            # Just log training metrics when not validating
            epoch_time = time.time() - epoch_start_time
            
            # Get GPU metrics
            gpu_util = np.mean(gpu_metrics.metrics['utilization'][-100:]) if gpu_metrics.metrics['utilization'] else 0
            gpu_mem = np.mean(gpu_metrics.metrics['memory_allocated'][-100:]) if gpu_metrics.metrics['memory_allocated'] else 0
            gpu_mem_peak = gpu_metrics.max_memory_allocated
            
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},,,,{gpu_util:.1f},{gpu_mem:.1f},{gpu_mem_peak:.1f}\n")
            
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            logger.info(f"Training loss: {train_loss:.4f}")
            logger.info(f"GPU memory - Current: {gpu_mem:.1f}MB, Peak: {gpu_mem_peak:.1f}MB")
        
        # Clean up memory before checkpoint saving
        torch.cuda.empty_cache()
        
        # Save regular checkpoints
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss if 'val_loss' in locals() else float('nan'),
                    'abs_rel': abs_rel if 'abs_rel' in locals() else float('nan'),
                    'rmse': rmse if 'rmse' in locals() else float('nan'),
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                # Try to save with just the essential data
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, os.path.join(output_dir, 'checkpoints', f'model_minimal_epoch_{epoch+1}.pth'))
                    logger.info(f"Saved minimal checkpoint at epoch {epoch+1}")
                except Exception as e2:
                    logger.error(f"Error saving minimal checkpoint: {e2}")
        
        # Force cleanup at end of epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final test for quantization
    if hasattr(args, 'test_quantization') and args.test_quantization:
        logger.info("Running quantization test...")
        try:
            quantization_test(model, val_loader, device, num_samples=5, output_dir=output_dir)
        except Exception as e:
            logger.error(f"Error during quantization test: {e}")
    
    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_training_time))}")
    
    # Create plots of training progress
    try:
        create_training_plots(log_file, output_dir)
    except Exception as e:
        logger.error(f"Error creating training plots: {e}")
    
    # Stop GPU metrics tracking and save plots
    gpu_metrics.stop()
    gpu_metrics.plot_metrics(output_dir)
    
    # Final cleanup
    if hasattr(train_dataset, 'shutdown'):
        train_dataset.shutdown()
    
    if hasattr(val_dataset, 'shutdown'):
        val_dataset.shutdown()
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

def create_training_plots(log_file, output_dir):
    """Create plots of training metrics for analysis"""
    # Create plot directory
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        # Import pandas for data processing
        import pandas as pd
        
        # Read log file
        try:
            df = pd.read_csv(log_file)
        except Exception as e:
            logger.warning(f"Could not read log file {log_file} for plotting: {e}")
            return
        
        # Check if we have enough data
        if len(df) < 2:
            logger.warning("Not enough training data for plotting")
            return
        
        # Create loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in df.columns:
            # Filter out rows where val_loss is NaN
            val_df = df[df['val_loss'].notna()]
            if len(val_df) > 0:
                plt.plot(val_df['epoch'], val_df['val_loss'], 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'loss.png'), dpi=150)
        plt.close()
        
        # Create metrics plot if available
        if 'abs_rel' in df.columns and 'rmse' in df.columns:
            metrics_df = df[df['abs_rel'].notna()]
            if len(metrics_df) > 0:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Absolute Relative Error', color='blue')
                ax1.plot(metrics_df['epoch'], metrics_df['abs_rel'], 'b-', label='AbsRel')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('RMSE', color='red')
                ax2.plot(metrics_df['epoch'], metrics_df['rmse'], 'r-', label='RMSE')
                ax2.tick_params(axis='y', labelcolor='red')
                
                plt.title('Validation Metrics')
                
                # Add combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plot_dir, 'metrics.png'), dpi=150)
                plt.close()
        
        # Create learning rate plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['epoch'], df['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'learning_rate.png'), dpi=150)
        plt.close()
        
        # Create GPU memory usage plot if available
        if 'gpu_mem_used' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if 'gpu_mem_peak' in df.columns:
                plt.plot(df['epoch'], df['gpu_mem_peak'], 'r-', label='Peak Memory')
            
            plt.plot(df['epoch'], df['gpu_mem_used'], 'b-', label='Avg Memory')
            plt.xlabel('Epoch')
            plt.ylabel('GPU Memory (MB)')
            plt.title('GPU Memory Usage During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add 8GB line for GTX 1080 reference
            plt.axhline(y=8*1024, color='gray', linestyle='--', alpha=0.5, label='8GB Limit')
            
            plt.savefig(os.path.join(plot_dir, 'epoch_gpu_memory.png'), dpi=150)
            plt.close()
        
        # Create combined metrics plot
        try:
            # Filter for epochs with all metrics
            combined_df = df.dropna(subset=['train_loss', 'val_loss', 'abs_rel', 'rmse'])
            
            if len(combined_df) >= 2:
                # Create plot with 3 subplots
                fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                
                # Loss plot
                axs[0].plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
                axs[0].plot(combined_df['epoch'], combined_df['val_loss'], 'r-', label='Validation Loss')
                axs[0].set_ylabel('Loss')
                axs[0].set_title('Training and Validation Loss')
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)
                
                # Metrics plot
                axs[1].plot(combined_df['epoch'], combined_df['abs_rel'], 'm-', label='AbsRel')
                axs[1].plot(combined_df['epoch'], combined_df['rmse'], 'g-', label='RMSE')
                axs[1].set_ylabel('Metrics')
                axs[1].set_title('Validation Metrics')
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)
                
                # Learning rate plot
                axs[2].plot(df['epoch'], df['lr'], 'k-')
                axs[2].set_ylabel('Learning Rate')
                axs[2].set_xlabel('Epoch')
                axs[2].set_title('Learning Rate Schedule')
                axs[2].set_yscale('log')
                axs[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'combined_metrics.png'), dpi=150)
                plt.close()
        except Exception as e:
            logger.warning(f"Error creating combined plot: {e}")
        
        logger.info(f"Training plots saved to {plot_dir}")
        
    except Exception as e:
        logger.error(f"Error in create_training_plots: {e}")
    finally:
        # Always close all plots to prevent memory leaks
        plt.close('all')

class QATPrefetcher:
    """Memory-efficient prefetcher for better data loading performance on GTX 1080"""
    def __init__(self, loader, device="cuda"):
        self.loader = loader
        self.device = device
        self.loader_iter = iter(loader)
        self.next_batch = None
        self._preload()
    
    def _preload(self):
        """Preload next batch without transferring to GPU"""
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
        except Exception as e:
            logger.error(f"Error in prefetcher: {e}")
            self.next_batch = None
    
    def next(self):
        """Get next batch (stays on CPU until training_loop moves it)"""
        if self.next_batch is None:
            return None
            
        batch = self.next_batch
        self._preload()
        return batch
