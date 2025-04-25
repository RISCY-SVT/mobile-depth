import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os

class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant loss для задачи оценки глубины.
    Оптимизирована для работы с ограниченной видеопамятью.
    """
    def __init__(self, epsilon=1e-8, lamda=0.5):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = epsilon
        self.lamda = lamda
        
    def forward(self, pred, target):
        # Избегаем log(0) - больше защита от нулевых значений
        valid_mask = (target > self.epsilon) & (pred > self.epsilon)
        
        # Проверяем, есть ли достаточное количество валидных пикселей
        if not torch.any(valid_mask) or torch.sum(valid_mask) < 10:
            # Возвращаем простой L1 loss если недостаточно данных
            return F.l1_loss(pred, target)
        
        # Применяем маску более эффективно
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # Логарифмическая разница с оптимизацией вычислений
        diff = torch.log(pred_valid) - torch.log(target_valid)
        
        # L1 часть
        l1_loss = torch.mean(torch.abs(diff))
        
        # Вариационный член для инвариантности к масштабу
        if self.lamda > 0 and diff.numel() > 1:
            # Используем эффективное вычисление вариации
            var_term = torch.var(diff, unbiased=False)
            total_loss = l1_loss - self.lamda * torch.sqrt(var_term + 1e-10)
            return torch.abs(total_loss)  # Гарантируем положительное значение
        else:
            return l1_loss

class BerHuLoss(nn.Module):
    """
    Обратная функция Хьюбера (Reverse Huber/BerHu) - оптимизирована для
    задач глубины и устойчивости к квантизации с учетом ограниченной памяти.
    """
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
        # Защита от экстремальных значений для стабильности квантизации
        pred = torch.clamp(pred, min=1e-6) 
        target = torch.clamp(target, min=1e-6)
        
        # Вычисляем абсолютную ошибку эффективно
        abs_diff = torch.abs(pred - target)
        
        # Разделяем тензор по порогу для эффективной обработки
        mask = abs_diff <= self.threshold
        
        # Используем где возможно встроенные операции PyTorch
        l1_part = abs_diff[mask].mean() if torch.any(mask) else torch.tensor(0.0, device=pred.device)
        
        # Проверяем, есть ли значения больше порога
        if torch.any(~mask):
            # L2 часть с оптимизированным вычислением
            l2_part = abs_diff[~mask]
            l2_part = ((l2_part**2 + self.threshold**2) / (2 * self.threshold)).mean()
            
            # Взвешенная сумма вместо конкатенации
            mask_ratio = mask.float().mean()
            loss = mask_ratio * l1_part + (1 - mask_ratio) * l2_part
        else:
            loss = l1_part
            
        return loss

class GradientLoss(nn.Module):
    """
    Оптимизированная потеря градиента для лучшей согласованности
    и плавности карты глубины. Переработана для уменьшения потребления памяти.
    """
    def __init__(self, weight=1.0, edge_threshold=0.05):
        super(GradientLoss, self).__init__()
        self.weight = weight
        self.edge_threshold = edge_threshold
        
    def forward(self, pred, target):
        # Эффективное вычисление градиентов с меньшим потреблением памяти
        # По горизонтали (x)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        
        # Освобождаем немного памяти при необходимости
        if pred.shape[2] * pred.shape[3] > 320 * 320:
            grad_diff_x = torch.abs(pred_dx - target_dx)
            edge_mask_x = target_dx > self.edge_threshold
            
            # Обработка краев для X с учетом памяти
            if torch.any(edge_mask_x):
                edge_loss_x = grad_diff_x[edge_mask_x].mean() * 2.0
                # Высвобождаем память
                del grad_diff_x
                del edge_mask_x
                torch.cuda.empty_cache()
            else:
                edge_loss_x = 0.0
                del grad_diff_x
                del edge_mask_x
                torch.cuda.empty_cache()
        else:
            # Обычная обработка для меньших размеров
            grad_diff_x = torch.abs(pred_dx - target_dx)
            edge_mask_x = target_dx > self.edge_threshold
            edge_loss_x = grad_diff_x[edge_mask_x].mean() * 2.0 if torch.any(edge_mask_x) else 0.0
        
        # По вертикали (y)
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        if pred.shape[2] * pred.shape[3] > 320 * 320:
            grad_diff_y = torch.abs(pred_dy - target_dy)
            edge_mask_y = target_dy > self.edge_threshold
            
            if torch.any(edge_mask_y):
                edge_loss_y = grad_diff_y[edge_mask_y].mean() * 2.0
                del grad_diff_y
                del edge_mask_y
                torch.cuda.empty_cache()
            else:
                edge_loss_y = 0.0
                del grad_diff_y
                del edge_mask_y
                torch.cuda.empty_cache()
        else:
            grad_diff_y = torch.abs(pred_dy - target_dy)
            edge_mask_y = target_dy > self.edge_threshold
            edge_loss_y = grad_diff_y[edge_mask_y].mean() * 2.0 if torch.any(edge_mask_y) else 0.0
        
        # Освобождаем память для градиентов после использования
        del pred_dx, target_dx, pred_dy, target_dy
        
        # Обычная градиентная потеря с оптимизированным потреблением памяти
        smooth_loss_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]).mean()
        smooth_loss_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]).mean()
        
        # Комбинируем потери
        return (smooth_loss_x + smooth_loss_y + edge_loss_x + edge_loss_y) * self.weight

class SmoothnessLoss(nn.Module):
    """
    Оптимизированная потеря сглаживания с учетом границ в изображении 
    и устойчивости к квантизации. Разделена на этапы для меньшего потребления памяти.
    """
    def __init__(self, weight=0.1, edge_factor=10.0):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight
        self.edge_factor = edge_factor
        
    def forward(self, depth, image):
        # Если изображение имеет 3 канала, приводим к одному для расчета градиентов
        if image.shape[1] > 1:
            image_gray = torch.mean(image, dim=1, keepdim=True)
        else:
            image_gray = image
        
        # Для больших изображений оптимизируем потребление памяти
        if depth.shape[2] * depth.shape[3] > 320 * 320:
            # Вычисляем градиенты глубины поэтапно
            depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
            image_dx = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
            
            # Вычисляем веса с контролируемым диапазоном
            weights_x = torch.exp(-image_dx * self.edge_factor)
            weights_x = torch.clamp(weights_x, 0.1, 1.0)
            
            # Финальная потеря с оптимизированным вычислением
            smoothness_x = (depth_dx * weights_x).sum() / (weights_x.sum() + 1e-7)
            
            # Очищаем память
            del depth_dx, image_dx, weights_x
            torch.cuda.empty_cache()
            
            # Вычисляем градиенты по Y аналогично
            depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
            image_dy = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])
            
            weights_y = torch.exp(-image_dy * self.edge_factor)
            weights_y = torch.clamp(weights_y, 0.1, 1.0)
            
            smoothness_y = (depth_dy * weights_y).sum() / (weights_y.sum() + 1e-7)
            
            # Очищаем память
            del depth_dy, image_dy, weights_y
            torch.cuda.empty_cache()
        else:
            # Стандартное вычисление для изображений меньшего размера
            depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
            depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
            
            image_dx = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
            image_dy = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])
            
            weights_x = torch.exp(-image_dx * self.edge_factor)
            weights_y = torch.exp(-image_dy * self.edge_factor)
            
            weights_x = torch.clamp(weights_x, 0.1, 1.0)
            weights_y = torch.clamp(weights_y, 0.1, 1.0)
            
            smoothness_x = (depth_dx * weights_x).mean() / (weights_x.mean() + 1e-7)
            smoothness_y = (depth_dy * weights_y).mean() / (weights_y.mean() + 1e-7)
        
        return (smoothness_x + smoothness_y) * self.weight

class SSIM(nn.Module):
    """
    Оптимизированная версия SSIM с низким потреблением памяти и опциональным
    пространственным даунскейлом для работы с ограниченной памятью GPU.
    """
    def __init__(self,
                 window_size: int = 7,
                 downscale_factor: int = 2,
                 C1: float = 0.01,
                 C2: float = 0.03) -> None:
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"

        self.w = window_size
        self.sigma = 1.5
        self.downscale = max(1, downscale_factor)
        
        # C1, C2 уже возведены в квадрат для стабилизации
        self.C1 = C1 ** 2
        self.C2 = C2 ** 2

        # Кешированное гауссово ядро (рассчитывается только один раз)
        _1d = torch.arange(self.w, dtype=torch.float32) - (self.w - 1) / 2
        g = torch.exp(-(_1d ** 2) / (2 * self.sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        window_2d = g @ g.t()
        
        # Регистрируем буфер, но не делаем его постоянным для экономии памяти
        self.register_buffer("_window_base", window_2d.unsqueeze(0).unsqueeze(0), persistent=False)
        self._kernel_cache = {}  # кеш для разных device/dtype
    
    def _get_window(self, ref_tensor):
        """Получить ядро свертки подходящего типа и на нужном устройстве"""
        device, dtype = ref_tensor.device, ref_tensor.dtype
        key = (device, dtype)
        
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._window_base.to(device=device, dtype=dtype)
        
        return self._kernel_cache[key]
    
    @staticmethod
    def _avg_pool(x, k):
        """Эффективное понижение разрешения с меньшим потреблением памяти"""
        if k == 1:
            return x
        return F.avg_pool2d(x, kernel_size=k, stride=k, padding=0, ceil_mode=False)

    def forward(self, pred, target):
        """
        pred, target : (N,1,H,W) – dtype может быть fp32 / fp16
        returns      : loss = 1-SSIM ∈ [0, 1]
        """
        # Адаптивно увеличиваем downscale для больших изображений 
        adaptive_downscale = self.downscale
        if pred.shape[2] * pred.shape[3] > 320 * 320:
            adaptive_downscale = max(self.downscale, 4)  # Увеличиваем для больших изображений
        
        # Понижение разрешения для экономии памяти
        if adaptive_downscale > 1:
            pred = self._avg_pool(pred, adaptive_downscale)
            target = self._avg_pool(target, adaptive_downscale)
        
        # Получаем подходящее ядро свертки
        window = self._get_window(pred)
        
        # Параметры для свертки
        pad = self.w // 2
        
        # Вычисляем средние значения с использованием свертки
        mu1 = F.conv2d(pred, window, padding=pad, groups=1)
        mu2 = F.conv2d(target, window, padding=pad, groups=1)
        
        # Вычисляем ковариацию и дисперсии поэтапно для экономии памяти
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Дисперсии и ковариация
        sigma1_sq = F.conv2d(pred * pred, window, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2
        
        # Вычисляем SSIM
        num = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        ssim_map = num / (den + 1e-12)
        
        # Возвращаем loss = 1 - mean(SSIM)
        return 1.0 - ssim_map.mean()

class EdgeAwareLoss(nn.Module):
    """
    Оптимизированная потеря для сохранения чётких границ глубины.
    Разделены этапы вычислений для снижения пикового потребления памяти.
    """
    def __init__(
        self,
        weight: float = 0.2,
        edge_threshold: float = 0.05,
        downscale_factor: int = 2,
    ):
        super().__init__()
        self.weight = weight
        self.edge_threshold = edge_threshold
        self.downscale_factor = downscale_factor
        
        # Параметры ядер Собеля
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)
        
        # Регистрируем как временные буферы
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Адаптивный даунскейл в зависимости от размера входных данных
        adaptive_downscale = self.downscale_factor
        if pred.shape[2] * pred.shape[3] > 320 * 320:
            adaptive_downscale = max(self.downscale_factor, 4)  # Больше даунскейл для больших изображений
        
        # Понижаем разрешение для снижения пиковой нагрузки на память
        if adaptive_downscale > 1:
            pred_ds = F.avg_pool2d(pred, adaptive_downscale, adaptive_downscale)
            target_ds = F.avg_pool2d(target, adaptive_downscale, adaptive_downscale)
        else:
            pred_ds, target_ds = pred, target
        
        # Получаем ядра на нужном устройстве
        kx = self.kx.to(pred_ds.device, pred_ds.dtype)
        ky = self.ky.to(pred_ds.device, pred_ds.dtype)
        
        # Эффективное вычисление градиентов
        ex_p = F.conv2d(pred_ds, kx, padding=1)
        ey_p = F.conv2d(pred_ds, ky, padding=1)
        
        ex_t = F.conv2d(target_ds, kx, padding=1)
        ey_t = F.conv2d(target_ds, ky, padding=1)
        
        # Вычисляем магнитуду градиентов
        mag_p = torch.sqrt(ex_p.pow(2) + ey_p.pow(2) + 1e-6)
        mag_t = torch.sqrt(ex_t.pow(2) + ey_t.pow(2) + 1e-6)
        
        # Определяем значимые границы
        mask = mag_t > self.edge_threshold
        
        # Если значимых границ нет, возвращаем ноль
        if not torch.any(mask):
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Вычисляем ошибку по абсолютной величине
        edge_err = F.l1_loss(mag_p[mask], mag_t[mask])
        
        # Оптимизированное вычисление ориентации если есть достаточно памяти
        if pred_ds.shape[2] * pred_ds.shape[3] <= 160 * 160:
            nx_p, ny_p = ex_p / (mag_p + 1e-7), ey_p / (mag_p + 1e-7)
            nx_t, ny_t = ex_t / (mag_t + 1e-7), ey_t / (mag_t + 1e-7)
            
            # Ошибка ориентации (косинусное расстояние)
            angle_err = 1 - torch.abs((nx_p * nx_t + ny_p * ny_t)[mask]).mean()
            return (edge_err + 0.5 * angle_err) * self.weight
        else:
            # Для больших изображений вычисляем только ошибку магнитуды
            return edge_err * self.weight

class RobustLoss(nn.Module):
    """
    Оптимизированная комбинация функций потерь для лучшей стабильности
    квантизированных моделей. Разделена на подзадачи для меньшего использования памяти.
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1, edge_weight=0.15):
        super(RobustLoss, self).__init__()
        self.alpha = alpha  # вес для L1
        self.beta = beta    # вес для BerHu
        self.gamma = gamma  # вес для Scale-Invariant
        self.delta = delta  # вес для SSIM
        self.edge_weight = edge_weight  # вес для Edge-Aware потери
        
        # Инициализируем компоненты потерь
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.berhu_loss = BerHuLoss(threshold=0.2)
        self.scale_invariant = ScaleInvariantLoss()
        self.ssim_loss = SSIM(window_size=7, downscale_factor=2)
        self.edge_loss = EdgeAwareLoss(
                            weight=edge_weight,     # Вес для границ
                            edge_threshold=0.05,    # Порог для границ
                            downscale_factor=2      # Уменьшаем разрешение для экономии памяти
                          )
        
        # Директория для визуализаций
        self.vis_dir = None
    
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Создает визуализацию для анализа работы функции потерь с защитой от OOM"""
        try:
            if self.vis_dir is None:
                self.vis_dir = output_dir
                os.makedirs(self.vis_dir, exist_ok=True)
            
            # Перемещаем данные на CPU перед визуализацией
            if isinstance(pred, list):
                pred_item = pred[0][0, 0].detach().cpu().numpy()
            else:
                pred_item = pred[0, 0].detach().cpu().numpy()
            
            target_item = target[0, 0].detach().cpu().numpy()
            rgb_item = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Освобождаем память GPU
            torch.cuda.empty_cache()
            
            # Денормализуем RGB
            rgb_item = rgb_item * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_item = np.clip(rgb_item, 0, 1)
            
            # Создаем карту ошибок
            error_map = np.abs(pred_item - target_item)
            
            # Создаем Sobel фильтры на CPU для экономии памяти GPU
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            # Применяем свертку на CPU для визуализации
            from scipy.ndimage import convolve
            edge_x_pred = convolve(pred_item, sobel_x, mode='reflect')
            edge_y_pred = convolve(pred_item, sobel_y, mode='reflect')
            edge_pred = np.sqrt(edge_x_pred**2 + edge_y_pred**2)
            
            edge_x_target = convolve(target_item, sobel_x, mode='reflect')
            edge_y_target = convolve(target_item, sobel_y, mode='reflect')
            edge_target = np.sqrt(edge_x_target**2 + edge_y_target**2)
            
            # Создаем визуализацию
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(rgb_item)
            plt.title('RGB Input')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(target_item, cmap='plasma')
            plt.title('Ground Truth Depth')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(pred_item, cmap='plasma')
            plt.title('Predicted Depth')
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            plt.imshow(error_map, cmap='hot')
            plt.title('Absolute Error')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            plt.imshow(edge_target, cmap='gray')
            plt.title('GT Edges')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            plt.imshow(edge_pred, cmap='gray')
            plt.title('Pred Edges')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, f'loss_vis_e{epoch}_b{batch_idx}.png'))
            plt.close('all')  # Явно закрываем все фигуры для освобождения памяти
            
            # Очищаем память
            del edge_x_pred, edge_y_pred, edge_pred, edge_x_target, edge_y_target, edge_target
            del pred_item, target_item, rgb_item, error_map
            
        except Exception as e:
            print(f"Error during visualization: {e}")
        finally:
            # Гарантированная очистка памяти
            torch.cuda.empty_cache()
        
    def forward(self, pred, target, rgb=None):
        # Для multi-scale выходов используем только основной выход
        if isinstance(pred, list):
            pred_solo = pred[0]
        else:
            pred_solo = pred
        
        # Стабилизация входных данных для численной устойчивости
        pred_solo = torch.clamp(pred_solo, 1e-3, 1.0 - 1e-3)
        target = torch.clamp(target, 1e-3, 1.0 - 1e-3)
        
        # Адаптивный подход к вычислению потерь в зависимости от размера
        is_large_image = pred_solo.shape[2] > 320 or pred_solo.shape[3] > 320
        
        # Вычисляем основные потери
        l1_loss = self.l1_loss(pred_solo, target)
        
        if is_large_image:
            # Для больших изображений используем упрощенный набор потерь
            berhu_loss = self.berhu_loss(pred_solo, target)
            total_loss = self.alpha * l1_loss + self.beta * berhu_loss
            
            # Edge-aware потери если доступно RGB изображение и оно не слишком большое
            if rgb is not None and self.edge_weight > 0:
                edge_aware = self.edge_loss(pred_solo, target)
                total_loss += self.edge_weight * edge_aware
                
            return total_loss
        else:
            # Используем полный набор потерь для изображений стандартного размера
            berhu_loss = self.berhu_loss(pred_solo, target)
            si_loss = self.scale_invariant(pred_solo, target)
            ssim_loss = self.ssim_loss(pred_solo, target)
            
            total_loss = (self.alpha * l1_loss + 
                        self.beta * berhu_loss + 
                        self.gamma * si_loss + 
                        self.delta * ssim_loss)
            
            # Edge-aware потери если доступно RGB изображение
            if rgb is not None and self.edge_weight > 0:
                edge_aware = self.edge_loss(pred_solo, target)
                total_loss += self.edge_weight * edge_aware
            
            return total_loss

class DepthWithSmoothnessLoss(nn.Module):
    """
    Оптимизированная комбинированная функция потерь для карт глубины,
    адаптированная для работы с ограниченной видеопамятью GTX 1080.
    """
    def __init__(self, base_weight=0.7, smoothness_weight=0.15, gradient_weight=0.15):
        super(DepthWithSmoothnessLoss, self).__init__()
        self.base_weight = base_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
        
        # Используем оптимизированные компоненты потерь
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(window_size=7, downscale_factor=2, C1=0.01, C2=0.03)
        self.smoothness_loss = SmoothnessLoss(weight=1.0, edge_factor=10.0)
        self.gradient_loss = GradientLoss(weight=1.0, edge_threshold=0.05)
        
        # Директория для визуализаций
        self.vis_dir = None
    
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Создает визуализацию для анализа с минимальным использованием GPU памяти"""
        try:
            if self.vis_dir is None:
                self.vis_dir = output_dir
                os.makedirs(self.vis_dir, exist_ok=True)
            
            # Перемещаем данные на CPU перед визуализацией
            if isinstance(pred, list):
                pred_item = pred[0][0, 0].detach().cpu().numpy()
            else:
                pred_item = pred[0, 0].detach().cpu().numpy()
            
            target_item = target[0, 0].detach().cpu().numpy()
            rgb_item = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Освобождаем память GPU
            torch.cuda.empty_cache()
            
            # Денормализуем RGB
            rgb_item = rgb_item * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_item = np.clip(rgb_item, 0, 1)
            
            # Создаем карту ошибок
            error_map = np.abs(pred_item - target_item)
            
            # Создаем и применяем Sobel фильтры на CPU
            from scipy.ndimage import convolve
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            edge_x_pred = convolve(pred_item, sobel_x, mode='reflect')
            edge_y_pred = convolve(pred_item, sobel_y, mode='reflect')
            edge_pred = np.sqrt(edge_x_pred**2 + edge_y_pred**2)
            
            edge_x_target = convolve(target_item, sobel_x, mode='reflect')
            edge_y_target = convolve(target_item, sobel_y, mode='reflect')
            edge_target = np.sqrt(edge_x_target**2 + edge_y_target**2)
            
            # Создаем визуализацию
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(rgb_item)
            plt.title('RGB Input')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(target_item, cmap='plasma')
            plt.title('Ground Truth Depth')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(pred_item, cmap='plasma')
            plt.title('Predicted Depth')
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            plt.imshow(error_map, cmap='hot')
            plt.title('Absolute Error')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            plt.imshow(edge_target, cmap='gray')
            plt.title('GT Edges')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            plt.imshow(edge_pred, cmap='gray')
            plt.title('Pred Edges')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, f'loss_vis_e{epoch}_b{batch_idx}.png'))
            plt.close('all')
            
            # Очищаем все переменные и освобождаем память
            del edge_x_pred, edge_y_pred, edge_pred, edge_x_target, edge_y_target, edge_target
            del pred_item, target_item, rgb_item, error_map, sobel_x, sobel_y
            
        except Exception as e:
            print(f"Error during visualization: {e}")
        finally:
            # Гарантированная очистка памяти
            torch.cuda.empty_cache()
    
    def forward(self, pred, target, rgb=None):
        # Для multi-scale выходов используем только основной выход
        if isinstance(pred, list):
            pred_solo = pred[0]
        else:
            pred_solo = pred
            
        # Стабилизация входных данных для лучшей численной устойчивости
        pred_solo = torch.clamp(pred_solo, 0.001, 0.999)
        target = torch.clamp(target, 0.001, 0.999)
        
        # Адаптивный подход в зависимости от размера входных данных
        use_simplified = pred_solo.shape[2] > 320 or pred_solo.shape[3] > 320
        
        # Вычисляем базовую потерю (L1 + опционально SSIM)
        l1_loss = self.l1_loss(pred_solo, target)
        
        if use_simplified:
            # Упрощенный вариант для больших изображений - только L1 для экономии памяти
            base_loss = l1_loss
        else:
            # Полный вариант с SSIM для стандартных размеров
            ssim_loss = self.ssim_loss(pred_solo, target)
            base_loss = 0.6 * l1_loss + 0.4 * ssim_loss
        
        # Опциональные потери сглаживания и градиентов
        smooth_loss = 0.0
        gradient_loss = 0.0
        
        if rgb is not None and not use_simplified:
            # Вычисляем потери сглаживания и градиентов для стандартных размеров
            smooth_loss = self.smoothness_loss(pred_solo, rgb)
            gradient_loss = self.gradient_loss(pred_solo, target)
        
        # Формируем итоговую потерю
        total_loss = self.base_weight * base_loss
        
        if smooth_loss > 0:
            total_loss += self.smoothness_weight * smooth_loss
            
        if gradient_loss > 0:
            total_loss += self.gradient_weight * gradient_loss
            
        return total_loss

class MultiScaleLoss(nn.Module):
    """
    Оптимизированная функция потерь для работы с multi-scale выходами модели,
    адаптированная для работы с ограниченной видеопамятью GTX 1080.
    """
    def __init__(self, base_criterion=None, weights=None):
        super(MultiScaleLoss, self).__init__()
        # Инициализируем базовый критерий
        self.base_criterion = base_criterion or RobustLoss(
            alpha=0.4,     # L1
            beta=0.3,      # BerHu
            gamma=0.2,     # Scale-Invariant
            delta=0.1,     # SSIM
            edge_weight=0.15  # Edge-Aware
        )
        
        # Веса для разных масштабов (убывают для меньших масштабов)
        self.weights = weights or [0.6, 0.2, 0.1, 0.1]
        
        # Визуализация делегируется базовому критерию
        self.vis_dir = None
    
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Делегирует визуализацию базовому критерию"""
        if hasattr(self.base_criterion, '_visualize_batch'):
            self.base_criterion._visualize_batch(pred, target, rgb, batch_idx, epoch, output_dir)
    
    def forward(self, pred, target, rgb=None):
        # Оптимизация для обработки мультимасштабных выходов
        if isinstance(pred, list):
            batch_size = pred[0].shape[0]
            num_scales = len(pred)
            
            # Слишком большой батч или изображение - используем только основной масштаб
            if batch_size > 8 or pred[0].shape[2] > 320 or pred[0].shape[3] > 320:
                # Используем только наибольший масштаб для экономии памяти
                if rgb is not None:
                    return self.base_criterion(pred[0], target, rgb)
                else:
                    return self.base_criterion(pred[0], target)
            
            # Стандартная обработка для небольших батчей и изображений
            total_loss = 0
            
            # Нормализуем веса к сумме 1
            weights = self.weights
            if len(weights) != num_scales:
                weights = [1.0/num_scales] * num_scales
            
            sum_weights = sum(weights)
            weights = [w/sum_weights for w in weights]
            
            # Вычисляем потери для каждого масштаба
            for i, pred_i in enumerate(pred):
                # Интерполируем предсказание, если размеры не совпадают
                if pred_i.shape[2:] != target.shape[2:]:
                    # Более эффективная билинейная интерполяция
                    pred_i = F.interpolate(
                        pred_i, 
                        size=target.shape[2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Вычисляем потерю
                if rgb is not None:
                    loss_i = self.base_criterion(pred_i, target, rgb)
                else:
                    loss_i = self.base_criterion(pred_i, target)
                
                total_loss += weights[i] * loss_i
                
                # Освобождаем память после каждого масштаба если это большие данные
                if i < num_scales - 1 and (pred_i.shape[2] > 160 or pred_i.shape[3] > 160):
                    del pred_i
                    torch.cuda.empty_cache()
            
            return total_loss
        else:
            # Один выход - просто используем базовый критерий
            if rgb is not None:
                return self.base_criterion(pred, target, rgb)
            else:
                return self.base_criterion(pred, target)
