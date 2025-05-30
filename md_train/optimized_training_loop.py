import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from threading import Thread
from fix_data_loader import create_single_process_dataloader
from depth_loss import MultiScaleLoss, DepthWithSmoothnessLoss

class GPUMetrics:
    """Class to track GPU metrics in a background thread"""
    def __init__(self, device, interval=1.0):
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
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self._monitor_gpu)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def _monitor_gpu(self):
        try:
            while self.running:
                if self.device.type == 'cuda':
                    self.metrics['timestamp'].append(time.time())
                    self.metrics['memory_allocated'].append(
                        torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
                    )
                    self.metrics['memory_reserved'].append(
                        torch.cuda.memory_reserved(self.device) / (1024**2)  # MB
                    )
                    
                    # Try to get GPU utilization if on Linux
                    try:
                        import subprocess
                        output = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                            encoding='utf-8'
                        )
                        self.metrics['utilization'].append(float(output.strip()))
                    except:
                        self.metrics['utilization'].append(0)
                
                time.sleep(self.interval)
        except Exception as e:
            print(f"GPU monitoring error: {e}")
    
    def plot_metrics(self, output_dir):
        if not self.metrics['timestamp']:
            print("No GPU metrics collected")
            return
            
        # Normalize timestamps
        start_time = self.metrics['timestamp'][0]
        timestamps = [(t - start_time) / 60.0 for t in self.metrics['timestamp']]  # minutes
        
        # Create plot directory
        plot_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create memory plot
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, self.metrics['memory_allocated'], label='Allocated Memory (MB)')
        plt.plot(timestamps, self.metrics['memory_reserved'], label='Reserved Memory (MB)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'gpu_memory.png'))
        plt.close('all')  # Explicitly close all figures to prevent memory leaks
        
        # Create utilization plot if available
        if any(self.metrics['utilization']):
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, self.metrics['utilization'], label='GPU Utilization (%)')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Utilization (%)')
            plt.title('GPU Utilization During Training')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'gpu_utilization.png'))
            plt.close('all')  # Explicitly close all figures to prevent memory leaks

def optimize_cuda_settings():
    """Apply optimized CUDA settings for faster training"""
    # Set CUDA Backend to run with maximum performance
    torch.backends.cudnn.benchmark = True
    # Enable TF32 precision for faster processing on newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Check if we're running on Windows
    if os.name == 'nt':
        # On Windows, set process priority to high
        try:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("Process priority set to HIGH")
        except:
            print("Failed to set process priority")


def train_epoch_optimized(model, train_loader, criterion, optimizer, scaler, device, 
                       mixed_precision=False, max_grad_norm=None, progress_hook=None):
    """Стабильная версия функции train_epoch_optimized с дополнительными проверками безопасности"""
    torch.cuda.empty_cache()
    
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    
    # Используем tqdm для отображения прогресса
    progress_bar = tqdm(train_loader, desc="Training")
    
    # Отслеживаем статистику
    batch_times = []
    forward_times = []
    backward_times = []
    data_times = []
    last_time = time.time()
    
    # Проверка на наличие NaN и Inf в параметрах модели
    for name, param in model.named_parameters():
        if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
            print(f"WARNING: Parameter {name} contains NaN or Inf values before training.")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Засекаем время загрузки данных
            data_time = time.time() - last_time
            data_times.append(data_time)
            
            # Явно переносим данные на GPU с синхронизацией
            rgb = batch['rgb'].to(device)
            depth_gt = batch['depth'].to(device)
            
            # Явная синхронизация
            torch.cuda.synchronize()
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход с обработкой ошибок
            forward_start = time.time()
            
            try:
                if mixed_precision:
                    with torch.amp.autocast('cuda'):
                        depth_pred = model(rgb)
                        loss = criterion(depth_pred, depth_gt, rgb)
                else:
                    depth_pred = model(rgb)
                    loss = criterion(depth_pred, depth_gt, rgb)
                
                # Проверка на NaN/Inf в выходных данных и потере
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Loss is NaN or Inf at batch {batch_idx}")
                    # Пропускаем проблемный батч
                    continue
                    
            except RuntimeError as e:
                print(f"Error during forward pass at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue
                
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Обратный проход с обработкой ошибок
            backward_start = time.time()
            
            try:
                if mixed_precision:
                    # Для mixed precision
                    scaler.scale(loss).backward()
                    
                    # Дополнительная синхронизация для стабильности
                    torch.cuda.synchronize()
                    
                    if max_grad_norm is not None:
                        # Отключаем градиенты, которые стали NaN/Inf
                        for param in model.parameters():
                            if param.grad is not None:
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Дополнительная синхронизация для стабильности
                    torch.cuda.synchronize()
                    
                    # Используем try/except для шага оптимизатора
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        print(f"Error during optimizer step at batch {batch_idx}: {e}")
                        # Сбрасываем скейлер, если возникла проблема
                        scaler._scale = torch.tensor(1.0, device=device)
                        torch.cuda.empty_cache()
                        continue
                        
                else:
                    # Стандартный проход без mixed precision
                    loss.backward()
                    
                    if max_grad_norm is not None:
                        # Отключаем градиенты, которые стали NaN/Inf
                        for param in model.parameters():
                            if param.grad is not None:
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
            
            except RuntimeError as e:
                print(f"Error during backward pass at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue
                
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Расчет времени всей итерации
            batch_time = time.time() - last_time
            batch_times.append(batch_time)
            last_time = time.time()
            
            # Обновляем статистику
            epoch_loss += loss.item()
            
            # Обновляем прогресс-бар
            num_samples = min(len(batch_times), 20)  # Берем среднее по последним 20 итерациям
            avg_forward = sum(forward_times[-num_samples:]) / num_samples
            avg_backward = sum(backward_times[-num_samples:]) / num_samples
            avg_data = sum(data_times[-num_samples:]) / num_samples
            avg_batch = sum(batch_times[-num_samples:]) / num_samples
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch': f"{avg_batch:.3f}s",
                'forward': f"{avg_forward:.3f}s",
                'backward': f"{avg_backward:.3f}s",
                'data': f"{avg_data:.3f}s"
            })
            
            # Вызываем хук прогресса, если он предоставлен
            if progress_hook:
                progress_hook(batch_idx, len(train_loader), loss.item())
            
            # Периодически очищаем кэш GPU для предотвращения утечек памяти
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Unexpected error at batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue
    
    progress_bar.close()
    
    # Рассчитываем средние значения для всей эпохи
    num_processed_batches = len(batch_times)
    if num_processed_batches > 0:
        avg_loss = epoch_loss / num_processed_batches
        avg_batch_time = sum(batch_times) / num_processed_batches
        avg_forward_time = sum(forward_times) / num_processed_batches
        avg_backward_time = sum(backward_times) / num_processed_batches
        avg_data_time = sum(data_times) / num_processed_batches
        
        # Выводим статистику
        print(f"Average times - Batch: {avg_batch_time:.3f}s, "
              f"Forward: {avg_forward_time:.3f}s, "
              f"Backward: {avg_backward_time:.3f}s, "
              f"Data: {avg_data_time:.3f}s")
    else:
        print("WARNING: No batches were successfully processed")
        avg_loss = float('inf')
    
    # Принудительная синхронизация и очистка кэша GPU
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    return avg_loss

class DataPrefetcher:
    """Prefetches data to GPU for faster data loading"""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_rgb = None
        self.next_depth = None
        self.next_batch = None
        self.preload()
    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            if 'rgb' in self.next_batch:
                self.next_batch['rgb'] = self.next_batch['rgb'].to(
                    self.device, non_blocking=True
                )
            
            if 'depth' in self.next_batch:
                self.next_batch['depth'] = self.next_batch['depth'].to(
                    self.device, non_blocking=True
                )
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

def validate_optimized_fixed(model, val_loader, criterion, device, output_dir=None, epoch=None):
    """Fixed validation function with proper CUDA synchronization and error handling"""
    # Reset CUDA state
    torch.cuda.empty_cache()
    model.eval()
    val_loss = 0
    abs_rel_error = 0
    rmse = 0
    
    
    # Use prefetcher for faster data loading
    progress_bar = tqdm(desc="Validation", total=len(val_loader))
    prefetcher = DataPrefetcher(val_loader, device)
    batch = prefetcher.next()
    batch_idx = 0
    
    with torch.no_grad():
        while batch is not None:
            try:
                # Explicit synchronization to ensure all previous CUDA operations complete
                torch.cuda.synchronize()
                
                # Get the data (already on device from prefetcher)
                rgb = batch['rgb']
                depth_gt = batch['depth']
                
                # Forward pass with explicit error handling
                depth_pred = model(rgb)
                
                # Another synchronization point
                torch.cuda.synchronize()
                
                # Calculate loss with careful handling of multi-scale outputs
                if isinstance(depth_pred, list):
                    depth_pred_solo = depth_pred[0]
                    loss = criterion(depth_pred_solo, depth_gt, rgb)
                else:
                    depth_pred_solo = depth_pred
                    loss = criterion(depth_pred, depth_gt, rgb)
                
                # Convert tensor to Python scalar with explicit synchronization
                torch.cuda.synchronize()
                val_loss += float(loss.item())  # Explicit conversion to Python float
                
                # Calculate metrics with explicit error handling
                try:
                    # Absolute relative error
                    rel_err = torch.abs(depth_pred_solo - depth_gt) / (depth_gt + 1e-6)
                    abs_rel_error += float(torch.mean(rel_err).item())
                    
                    # RMSE
                    rmse_batch = torch.sqrt(torch.mean((depth_pred_solo - depth_gt) ** 2))
                    rmse += float(rmse_batch.item())
                except Exception as e:
                    print(f"Warning: Error calculating metrics: {e}")
                    torch.cuda.empty_cache()  # Free memory used during visualization
                    # Continue even if metrics calculation fails
                
                # Visualization (only for first batch)
                if output_dir and batch_idx == 0 and epoch is not None:
                    try:
                        save_visualization(rgb, depth_gt, depth_pred_solo, output_dir, epoch)
                    except Exception as e:
                        print(f"Warning: Error saving visualization: {e}")
                        torch.cuda.empty_cache()  # Free memory used during visualization
                
                # Update progress bar
                progress_bar.update(1)
                
                # Get next batch with error handling
                try:
                    batch = prefetcher.next()
                except Exception as e:
                    print(f"Error getting next batch: {e}")
                    torch.cuda.empty_cache()  # Free memory used during visualization
                    batch = None
                    
                batch_idx += 1
                
                # Periodically clear cache to prevent fragmentation
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                torch.cuda.empty_cache()  # Free memory used during visualization
                # Try to continue with next batch
                try:
                    batch = prefetcher.next()
                    batch_idx += 1
                except:
                    break
    
    progress_bar.close()
    
    # Final cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Calculate average metrics
    avg_loss = val_loss / max(1, batch_idx)
    avg_abs_rel = abs_rel_error / max(1, batch_idx)
    avg_rmse = rmse / max(1, batch_idx)
    
    return avg_loss, avg_abs_rel, avg_rmse

def save_visualization(rgb, depth_gt, depth_pred, output_dir, epoch):
    """Enhanced visualization with additional metrics"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process only the first item from the batch
    rgb_np = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
    depth_gt_np = depth_gt[0, 0].detach().cpu().numpy()
    depth_pred_np = depth_pred[0, 0].detach().cpu().numpy()
    
    # Denormalize RGB image
    rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    
    # Calculate error map
    error_map = np.abs(depth_pred_np - depth_gt_np)
    
    # Create detailed visualization
    plt.figure(figsize=(15, 10))
    
    # RGB input
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_np)
    plt.title('RGB Input')
    plt.axis('off')
    
    # Ground truth depth
    plt.subplot(2, 2, 2)
    plt.imshow(depth_gt_np, cmap='plasma')
    plt.title(f'Ground Truth Depth (min={depth_gt_np.min():.2f}, max={depth_gt_np.max():.2f})')
    plt.axis('off')
    plt.colorbar()
    
    # Predicted depth
    plt.subplot(2, 2, 3)
    plt.imshow(depth_pred_np, cmap='plasma')
    plt.title(f'Predicted Depth (min={depth_pred_np.min():.2f}, max={depth_pred_np.max():.2f})')
    plt.axis('off')
    plt.colorbar()
    
    # Error map
    plt.subplot(2, 2, 4)
    plt.imshow(error_map, cmap='hot')
    plt.title(f'Absolute Error (mean={error_map.mean():.4f}, max={error_map.max():.4f})')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'epoch_{epoch:03d}.png'), dpi=150)
    plt.close('all')  # Explicitly close all figures to avoid memory leaks
    torch.cuda.empty_cache()  # Free memory used during visualization

def train_optimized(args, model, train_dataset, val_dataset):
    """Optimized training function with enhanced monitoring and performance"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Apply CUDA optimizations
    optimize_cuda_settings()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = MultiScaleLoss(base_criterion=DepthWithSmoothnessLoss(base_weight=0.85, smoothness_weight=0.15))
    
    # Используем AdamW с весовым затуханием для лучшей квантизации
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # L2 регуляризация
        eps=1e-3  # Стабильность для квантизации
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, 
        min_lr=args.min_lr
    )
    
    # Set up gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
    
    # Set up data loaders with optimal settings for your hardware
    train_loader = create_single_process_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = create_single_process_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize GPU metrics tracking
    gpu_metrics = GPUMetrics(device)
    gpu_metrics.start()
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {start_epoch-1}")
    
    # Set up logging
    log_file = os.path.join(output_dir, 'logs', 'training_log.csv')
    with open(log_file, 'a') as f:
        if start_epoch == 0:  # Only write header if starting fresh
            f.write("epoch,time,lr,train_loss,val_loss,abs_rel,rmse,gpu_util,gpu_mem_used\n")
    
    print("Starting training...")
    training_start_time = time.time()
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{args.epochs}, LR: {current_lr:.6f}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Train one epoch
        train_loss = train_epoch_optimized(
            model, train_loader, criterion, optimizer, scaler, device, 
            mixed_precision=args.mixed_precision,
            max_grad_norm=args.clip_grad
        )
        
        # Validate if needed
        if (epoch + 1) % args.val_freq == 0:
            val_loss, abs_rel, rmse = validate_optimized_fixed(
                model, val_loader, criterion, device, output_dir, epoch
            )
            
            # Update scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Check for LR changes
            if optimizer.param_groups[0]['lr'] != current_lr:
                print(f"Learning rate changed from {current_lr:.6f} to {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save metrics
            epoch_time = time.time() - epoch_start_time
            
            # Get GPU metrics
            gpu_util = np.mean(gpu_metrics.metrics['utilization'][-100:]) if gpu_metrics.metrics['utilization'] else 0
            gpu_mem = np.mean(gpu_metrics.metrics['memory_allocated'][-100:]) if gpu_metrics.metrics['memory_allocated'] else 0
            
            # Log metrics
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},"
                        f"{val_loss:.4f},{abs_rel:.4f},{rmse:.4f},{gpu_util:.1f},{gpu_mem:.1f}\n")
            
            # Print summary
            print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, AbsRel: {abs_rel:.4f}, RMSE: {rmse:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'abs_rel': abs_rel,
                    'rmse': rmse,
                    'best_val_loss': best_val_loss
                }, best_model_path)
                print(f"Saved best model with validation loss {val_loss:.4f}")
        else:
            # Just log training metrics when not validating
            epoch_time = time.time() - epoch_start_time
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},,,,,\n")
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"Training loss: {train_loss:.4f}")
            
            # Update scheduler based on training loss when not validating
            scheduler.step(train_loss)
        
        # Save regular checkpoints
        torch.cuda.empty_cache()
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Training complete
    total_training_time = time.time() - training_start_time
    print(f"Training completed in {datetime.timedelta(seconds=int(total_training_time))}")
    
    # Stop GPU metrics tracking and save plots
    gpu_metrics.stop()
    gpu_metrics.plot_metrics(output_dir)
    
    # Final cleanup
    if hasattr(train_dataset, 'shutdown'):
        train_dataset.shutdown()
    
    if hasattr(val_dataset, 'shutdown'):
        val_dataset.shutdown()
    
    return model
