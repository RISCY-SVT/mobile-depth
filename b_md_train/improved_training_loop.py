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
import torch.nn.utils.prune as prune
import logging
import math

from improved_depth_loss import MultiScaleLoss, DepthWithSmoothnessLoss, RobustLoss, BerHuLoss
from improved_fix_data_loader import QATPrefetcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            logger.error(f"GPU monitoring error: {e}")
    
    def plot_metrics(self, output_dir):
        if not self.metrics['timestamp']:
            logger.warning("No GPU metrics collected")
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
            logger.info("Process priority set to HIGH")
        except:
            logger.warning("Failed to set process priority")

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
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
    
    logger.info(f"Applied temporary L1 pruning with amount {pruning_amount}")

def remove_weight_pruning(model):
    """Remove pruning and make weights permanent"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    logger.info("Removed pruning masks, pruned weights are now permanent")

def prepare_qat(model, num_bits=8):
    """
    Prepare model for Quantization-Aware Training (QAT).
    Важно делать это на поздних этапах обучения.
    """
    if not hasattr(torch.quantization, 'prepare_qat'):
        logger.warning("PyTorch version doesn't support prepare_qat. Skipping QAT preparation.")
        return model
    
    # Настройка quantization config
    qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    # Применяем QAT к модели
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    
    logger.info(f"Model prepared for QAT with {num_bits} bits")
    return model

def train_epoch_improved(model, train_loader, criterion, optimizer, scaler, device, 
                         epoch, max_epochs, mixed_precision=False, max_grad_norm=None, 
                         update_lr_fn=None, batch_visualize_freq=500):
    """Улучшенная функция обучения эпохи с множеством оптимизаций для устойчивости тренировки"""
    torch.cuda.empty_cache()
    
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    total_vis_loss = 0
    vis_batches = 0
    
    # Используем QATPrefetcher для эффективной загрузки данных
    prefetcher = QATPrefetcher(train_loader, device)
    
    # Используем tqdm для отображения прогресса
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epochs}")
    
    # Отслеживаем статистику
    batch_times = []
    forward_times = []
    backward_times = []
    data_times = []
    last_time = time.time()
    
    # Обновление learning rate, если необходимо
    if update_lr_fn is not None:
        current_lr = update_lr_fn(optimizer, epoch)
        logger.info(f"Set learning rate to {current_lr:.6f}")
    
    batch_idx = 0
    batch = prefetcher.next()
    
    while batch is not None:
        try:
            # Засекаем время загрузки данных
            data_time = time.time() - last_time
            data_times.append(data_time)
            
            # Явно переносим данные на GPU с синхронизацией
            rgb = batch['rgb']
            depth_gt = batch['depth']
            
            # Явная синхронизация
            torch.cuda.synchronize()
            
            # Обнуляем градиенты
            optimizer.zero_grad(set_to_none=True)  # более эффективное обнуление градиентов
            
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
                    logger.warning(f"WARNING: Loss is NaN or Inf at batch {batch_idx}. Skipping.")
                    batch_idx += 1
                    batch = prefetcher.next()
                    continue
                    
            except RuntimeError as e:
                logger.error(f"Error during forward pass at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                batch_idx += 1
                batch = prefetcher.next()
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
                        scaler.unscale_(optimizer)
                        for param in model.parameters():
                            if param.grad is not None:
                                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # Дополнительная синхронизация для стабильности
                    torch.cuda.synchronize()
                    
                    # Используем try/except для шага оптимизатора
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        logger.error(f"Error during optimizer step at batch {batch_idx}: {e}")
                        # Сбрасываем скейлер, если возникла проблема
                        scaler._scale = torch.tensor(1.0, device=device)
                        torch.cuda.empty_cache()
                        batch_idx += 1
                        batch = prefetcher.next()
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
                logger.error(f"Error during backward pass at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                batch_idx += 1
                batch = prefetcher.next()
                continue
                
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            
            # Расчет времени всей итерации
            batch_time = time.time() - last_time
            batch_times.append(batch_time)
            last_time = time.time()
            
            # Обновляем статистику
            epoch_loss += loss.item()
            
            # Вычисляем визуальную потерю (для вывода)
            total_vis_loss += loss.item()
            vis_batches += 1
            
            # Периодически сохраняем визуализацию промежуточных результатов
            # ###############################################################################
            if batch_idx % batch_visualize_freq == 0 and hasattr(criterion, '_visualize_batch'):
                criterion._visualize_batch(depth_pred, depth_gt, rgb, batch_idx, epoch)
            
            # Обновляем прогресс-бар
            num_samples = min(len(batch_times), 20)  # Берем среднее по последним 20 итерациям
            avg_forward = sum(forward_times[-num_samples:]) / num_samples
            avg_backward = sum(backward_times[-num_samples:]) / num_samples
            avg_data = sum(data_times[-num_samples:]) / num_samples
            avg_batch = sum(batch_times[-num_samples:]) / num_samples
            avg_loss = total_vis_loss / vis_batches
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'batch': f"{avg_batch:.3f}s",
                'forward': f"{avg_forward:.3f}s",
                'backward': f"{avg_backward:.3f}s",
                'data': f"{avg_data:.3f}s"
            })
            progress_bar.update(1)
            
            # Периодически очищаем кэш GPU для предотвращения утечек памяти
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            batch_idx += 1
            batch = prefetcher.next()
                
        except Exception as e:
            logger.error(f"Unexpected error at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            batch_idx += 1
            batch = prefetcher.next()
            continue
    
    progress_bar.close()
    
    # Рассчитываем средние значения для всей эпохи
    num_processed_batches = max(batch_idx, 1)
    avg_loss = epoch_loss / num_processed_batches
    
    # Вывод статистики
    if len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_backward_time = sum(backward_times) / len(backward_times)
        avg_data_time = sum(data_times) / len(data_times)
        
        logger.info(f"Average times - Batch: {avg_batch_time:.3f}s, "
              f"Forward: {avg_forward_time:.3f}s, "
              f"Backward: {avg_backward_time:.3f}s, "
              f"Data: {avg_data_time:.3f}s")
    
    # Принудительная синхронизация и очистка кэша GPU
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    return avg_loss

def validate_improved(model, val_loader, criterion, device, output_dir=None, epoch=None):
    """Улучшенная функция валидации с более аккуратной обработкой ошибок и мониторингом"""
    # Reset CUDA state
    torch.cuda.empty_cache()
    model.eval()
    val_loss = 0
    abs_rel_error = 0
    rmse = 0
    
    # Используем QATPrefetcher для эффективной загрузки данных
    prefetcher = QATPrefetcher(val_loader, device)
    
    # Use prefetcher for faster data loading
    progress_bar = tqdm(total=len(val_loader), desc="Validation")
    
    batch = prefetcher.next()
    batch_idx = 0
    
    visualized = False
    
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
                    loss = criterion(depth_pred, depth_gt, rgb)
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
                    logger.warning(f"Error calculating metrics: {e}")
                    torch.cuda.empty_cache()  # Free memory used during visualization
                    # Continue even if metrics calculation fails
                
                # Visualization (only for first batch if not already visualized)
                if output_dir and not visualized and epoch is not None:
                    try:
                        save_visualization(rgb, depth_gt, depth_pred_solo, output_dir, epoch)
                        visualized = True
                    except Exception as e:
                        logger.warning(f"Error saving visualization: {e}")
                        torch.cuda.empty_cache()  # Free memory used during visualization
                
                # Update progress bar
                progress_bar.update(1)
                
                # Get next batch with error handling
                try:
                    batch = prefetcher.next()
                except Exception as e:
                    logger.error(f"Error getting next batch: {e}")
                    torch.cuda.empty_cache()
                    batch = None
                    
                batch_idx += 1
                
                # Periodically clear cache to prevent fragmentation
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
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
    num_processed_batches = max(batch_idx, 1)
    avg_loss = val_loss / num_processed_batches
    avg_abs_rel = abs_rel_error / num_processed_batches
    avg_rmse = rmse / num_processed_batches
    
    return avg_loss, avg_abs_rel, avg_rmse

def save_visualization(rgb, depth_gt, depth_pred, output_dir, epoch):
    """Enhanced visualization with additional metrics and improved presentation"""
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
    
    # Calculate metrics for this visualization
    rel_error = np.mean(np.abs(depth_pred_np - depth_gt_np) / (depth_gt_np + 1e-6))
    rmse = np.sqrt(np.mean((depth_pred_np - depth_gt_np) ** 2))
    
    # Create detailed visualization with more information
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
    
    # Add overall title with epoch information
    plt.suptitle(f'Epoch {epoch+1} - B_MobileDepth Results', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png'), dpi=150)
    plt.close('all')  # Explicitly close all figures to avoid memory leaks
    torch.cuda.empty_cache()  # Free memory used during visualization

def quantization_test(model, test_loader, device, num_samples=10, output_dir=None):
    """
    Simulate quantization effects on the model and test it with a few samples.
    This helps diagnose potential issues before actual quantization.
    
    Adapted for B_MobileDepth model.
    """
    logger.info("Testing model with simulated quantization...")
    
    # Force FP32 for baseline results
    model = model.to(device).float().eval()
    
    # Prepare quantized version for simulation
    try:
        # Create a copy of the model for quantization simulation
        import copy
        model_quant = copy.deepcopy(model)
        
        # Configure quantization
        model_quant.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model_quant, inplace=True)
        
        # Convert to quantized model
        torch.quantization.convert(model_quant, inplace=True)
        
        # Create visualization directory
        if output_dir:
            quant_vis_dir = os.path.join(output_dir, 'quantization_test')
            os.makedirs(quant_vis_dir, exist_ok=True)
        
        # Test both models with a few samples
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Get data
            rgb = batch['rgb'].to(device)
            depth_gt = batch['depth'].to(device)
            
            # Original model inference
            with torch.no_grad():
                depth_pred_original = model(rgb)
                if isinstance(depth_pred_original, list):
                    depth_pred_original = depth_pred_original[0]
            
            # Quantized model inference (first convert to CPU since quantized model only works on CPU)
            with torch.no_grad():
                rgb_cpu = rgb.cpu()
                depth_pred_quant = model_quant(rgb_cpu)
                if isinstance(depth_pred_quant, list):
                    depth_pred_quant = depth_pred_quant[0]
                depth_pred_quant = depth_pred_quant.to(device)
            
            # Calculate metrics
            rel_err_original = torch.abs(depth_pred_original - depth_gt) / (depth_gt + 1e-6)
            rel_err_quant = torch.abs(depth_pred_quant - depth_gt) / (depth_gt + 1e-6)
            
            mean_rel_original = torch.mean(rel_err_original).item()
            mean_rel_quant = torch.mean(rel_err_quant).item()
            
            rmse_original = torch.sqrt(torch.mean((depth_pred_original - depth_gt) ** 2)).item()
            rmse_quant = torch.sqrt(torch.mean((depth_pred_quant - depth_gt) ** 2)).item()
            
            logger.info(f"Sample {i+1} - Original: RelErr={mean_rel_original:.4f}, RMSE={rmse_original:.4f}, "
                        f"Quantized: RelErr={mean_rel_quant:.4f}, RMSE={rmse_quant:.4f}")
            
            # Save visualization
            if output_dir:
                # Create enhanced visualization comparing original and quantized outputs
                rgb_np = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
                depth_gt_np = depth_gt[0, 0].detach().cpu().numpy()
                depth_pred_original_np = depth_pred_original[0, 0].detach().cpu().numpy()
                depth_pred_quant_np = depth_pred_quant[0, 0].detach().cpu().numpy()
                
                # Denormalize RGB
                rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                
                # Calculate error maps
                error_original = np.abs(depth_pred_original_np - depth_gt_np)
                error_quant = np.abs(depth_pred_quant_np - depth_gt_np)
                
                # Create visualization with more details
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
        
        logger.info("Quantization test completed.")
        return True
    
    except Exception as e:
        logger.error(f"Error during quantization test: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_improved(args, model, train_dataset, val_dataset):
    """
    Улучшенная функция тренировки с поддержкой multi-scale loss,
    оптимизациями под квантизацию и расширенным мониторингом.
    
    Адаптирована специально для B_MobileDepth модели.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Настраиваем файловый лог в дополнение к консольному
    file_handler = logging.FileHandler(os.path.join(output_dir, 'logs', 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Apply CUDA optimizations
    optimize_cuda_settings()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Move model to device
    model = model.to(device)
    
    # Set up improved loss function based on model output type
    if hasattr(model, 'training') and model.training and hasattr(args, 'use_robust_loss') and args.use_robust_loss:
        # Use MultiScaleLoss for multi-scale outputs with RobustLoss
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
    
    # Используем AdamW с весовым затуханием для лучшей квантизации
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # L2 регуляризация
        eps=1e-3  # Стабильность для квантизации
    )
    
    # Set up gradient scaler for mixed precision
    scaler = torch.amp.GradScaler() if args.mixed_precision else None
    
    # Set up data loaders with optimal settings
    from improved_fix_data_loader import create_improved_dataloader
    
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
        pin_memory=args.pin_memory
    )
    
    # Initialize GPU metrics tracking
    gpu_metrics = GPUMetrics(device)
    gpu_metrics.start()
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch-1}")
    
    # Set up learning rate scheduler with warmup
    num_warmup_epochs = 5 if args.epochs >= 20 else 2
    
    # Learning rate scheduler update function
    def update_lr(optimizer, epoch):
        return cosine_learning_rate_with_warmup(
            optimizer, epoch, 
            warmup_epochs=num_warmup_epochs, 
            max_epochs=args.epochs, 
            init_lr=args.lr, 
            min_lr=args.min_lr
        )
    
    # Set up logging
    log_file = os.path.join(output_dir, 'logs', 'training_log.csv')
    with open(log_file, 'a') as f:
        if start_epoch == 0:  # Only write header if starting fresh
            f.write("epoch,time,lr,train_loss,val_loss,abs_rel,rmse,gpu_util,gpu_mem_used\n")
    
    logger.info("Starting training...")
    training_start_time = time.time()
    
    # Variables for QAT
    qat_started = False
    qat_start_epoch = int(args.epochs * 0.75)  # Start QAT at 75% of training
    pruning_percentage = 0.3  # Initial pruning percentage
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}, LR: {current_lr:.6f}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Quantization-Aware Training at later stages
        if epoch >= qat_start_epoch and not qat_started and hasattr(args, 'use_qat') and args.use_qat:
            logger.info(f"Starting Quantization-Aware Training at epoch {epoch+1}")
            model = prepare_qat(model)
            qat_started = True
        
        # Apply temporary weight pruning to simulate quantization effects (scaling pruning percentage)
        if epoch > 0 and epoch % 5 == 0 and pruning_percentage < 0.5:
            # Gradually increase pruning percentage to prepare the model
            pruning_percentage += 0.05
            apply_weight_pruning(model, pruning_percentage)
            # Train for one iteration with pruning
            train_epoch_improved(model, train_loader, criterion, optimizer, scaler, device, 
                         epoch, args.epochs, mixed_precision=args.mixed_precision,
                         max_grad_norm=args.clip_grad, update_lr_fn=update_lr)
            # Remove pruning masks but keep weights pruned
            remove_weight_pruning(model)
            
        # Train one epoch
        train_loss = train_epoch_improved(
            model, train_loader, criterion, optimizer, scaler, device, 
            epoch, args.epochs, mixed_precision=args.mixed_precision,
            max_grad_norm=args.clip_grad, update_lr_fn=update_lr
        )
        
        # Validate if needed
        if (epoch + 1) % args.val_freq == 0:
            val_loss, abs_rel, rmse = validate_improved(
                model, val_loader, criterion, device, output_dir, epoch
            )
            
            # Echo metrics
            epoch_time = time.time() - epoch_start_time
            
            # Get GPU metrics
            gpu_util = np.mean(gpu_metrics.metrics['utilization'][-100:]) if gpu_metrics.metrics['utilization'] else 0
            gpu_mem = np.mean(gpu_metrics.metrics['memory_allocated'][-100:]) if gpu_metrics.metrics['memory_allocated'] else 0
            
            # Log metrics
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},"
                        f"{val_loss:.4f},{abs_rel:.4f},{rmse:.4f},{gpu_util:.1f},{gpu_mem:.1f}\n")
            
            # Print summary
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            logger.info(f"Training loss: {train_loss:.4f}")
            logger.info(f"Validation - Loss: {val_loss:.4f}, AbsRel: {abs_rel:.4f}, RMSE: {rmse:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'abs_rel': abs_rel,
                    'rmse': rmse,
                    'best_val_loss': best_val_loss
                }, best_model_path)
                logger.info(f"Saved best model with validation loss {val_loss:.4f}")
        else:
            # Just log training metrics when not validating
            epoch_time = time.time() - epoch_start_time
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{epoch_time:.1f},{current_lr:.6f},{train_loss:.4f},,,,,\n")
            
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            logger.info(f"Training loss: {train_loss:.4f}")
        
        # Save regular checkpoints
        torch.cuda.empty_cache()
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Final test for quantization
    if hasattr(args, 'test_quantization') and args.test_quantization:
        logger.info("Running quantization test...")
        quantization_test(model, val_loader, device, num_samples=5, output_dir=output_dir)
    
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
    
    return model

def create_training_plots(log_file, output_dir):
    """Create plots of training metrics for analysis"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    import pandas as pd
    # Read log file
    try:
        df = pd.read_csv(log_file)
    except:
        logger.warning(f"Could not read log file {log_file} for plotting")
        return
    
    # Create loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    if 'val_loss' in df.columns:
        # Filter out rows where val_loss is NaN
        val_df = df[df['val_loss'].notna()]
        plt.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.close()
    
    # Create metrics plot if available
    if 'abs_rel' in df.columns and 'rmse' in df.columns:
        metrics_df = df[df['abs_rel'].notna()]
        if len(metrics_df) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Absolute Relative Error', color='tab:blue')
            ax1.plot(metrics_df['epoch'], metrics_df['abs_rel'], color='tab:blue', label='AbsRel')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('RMSE', color='tab:red')
            ax2.plot(metrics_df['epoch'], metrics_df['rmse'], color='tab:red', label='RMSE')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title('Validation Metrics')
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'metrics.png'))
            plt.close()
    
    # Create learning rate plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'learning_rate.png'))
    plt.close()
    