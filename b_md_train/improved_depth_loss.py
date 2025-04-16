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
    
    Комбинирует L1 потерю с регуляризацией масштаба для лучшей устойчивости
    и инвариантности к масштабу глубины.
    """
    def __init__(self, epsilon=1e-8, lamda=0.5):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = epsilon
        self.lamda = lamda
        
    def forward(self, pred, target):
        # Избегаем log(0)
        valid_mask = (target > self.epsilon) & (pred > self.epsilon)
        
        # Если нет валидных пикселей или их очень мало, возвращаем простой L1 loss
        if not torch.any(valid_mask) or torch.sum(valid_mask) < 10:
            return F.l1_loss(pred, target)
        
        # Применяем маску
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # Логарифмическая разница
        diff = torch.log(pred_valid) - torch.log(target_valid)
        
        # L1 часть
        l1_loss = torch.mean(torch.abs(diff))
        
        # Вариационный член для инвариантности к масштабу, только если у нас достаточно элементов
        if self.lamda > 0 and diff.numel() > 1:
            # Используем unbiased=False для избежания предупреждения
            var_term = torch.var(diff, unbiased=False)
            total_loss = l1_loss - self.lamda * torch.sqrt(var_term + 1e-10)
            return torch.abs(total_loss)  # Гарантируем положительное значение
        else:
            return l1_loss

class BerHuLoss(nn.Module):
    """
    Обратная функция Хьюбера (Reverse Huber/BerHu) - отлично подходит для задач глубины,
    особенно когда нужен баланс между обработкой малых и больших ошибок.
    Оптимизирована для устойчивости к квантизации.
    """
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
        # Гарантируем положительные значения для избежания проблем с квантизацией
        pred = torch.clamp(pred, min=1e-6) 
        target = torch.clamp(target, min=1e-6)
        
        # Вычисляем абсолютную ошибку
        abs_diff = torch.abs(pred - target)
        
        # Разделяем на случаи меньше и больше порога
        mask = abs_diff <= self.threshold
        
        # Для значений меньше порога используем L1
        l1_part = abs_diff[mask]
        
        # Проверяем, есть ли значения больше порога
        if torch.any(~mask):
            # Для значений больше порога используем L2
            l2_part = abs_diff[~mask]
            l2_part = (l2_part**2 + self.threshold**2) / (2 * self.threshold)
            
            # Объединяем результаты
            loss = torch.cat([l1_part, l2_part])
        else:
            loss = l1_part
            
        return loss.mean()

class GradientLoss(nn.Module):
    """
    Улучшенная потеря градиента с механизмами защиты от квантизационных артефактов
    для улучшения согласованности и плавности карты глубины.
    """
    def __init__(self, weight=1.0, edge_threshold=0.05):
        super(GradientLoss, self).__init__()
        self.weight = weight
        self.edge_threshold = edge_threshold
        
    def forward(self, pred, target):
        # Градиенты по x и y для предсказания (с защитой от резких изменений)
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        # Градиенты по x и y для ground truth
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # Ошибка градиентов с адаптивным взвешиванием для устойчивости к квантизации
        grad_diff_x = torch.abs(pred_dx - target_dx)
        grad_diff_y = torch.abs(pred_dy - target_dy)
        
        # Определяем края (большие градиенты в ground truth)
        edge_mask_x = target_dx > self.edge_threshold
        edge_mask_y = target_dy > self.edge_threshold
        
        # Усиливаем потери на границах для лучшего сохранения структуры
        if torch.any(edge_mask_x):
            edge_loss_x = grad_diff_x[edge_mask_x].mean() * 2.0
        else:
            edge_loss_x = 0.0
            
        if torch.any(edge_mask_y):
            edge_loss_y = grad_diff_y[edge_mask_y].mean() * 2.0
        else:
            edge_loss_y = 0.0
        
        # Обычная градиентная потеря
        smooth_loss_x = grad_diff_x.mean()
        smooth_loss_y = grad_diff_y.mean()
        
        # Комбинируем потери
        return (smooth_loss_x + smooth_loss_y + edge_loss_x + edge_loss_y) * self.weight

class SmoothnessLoss(nn.Module):
    """
    Улучшенная потеря сглаживания с учетом границ в изображении и устойчивости к квантизации.
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
            
        # Вычисляем градиенты глубины
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        # Вычисляем градиенты изображения
        image_dx = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
        image_dy = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])
        
        # Уменьшаем штраф, если есть края на изображении, с контролируемым диапазоном весов
        weights_x = torch.exp(-image_dx * self.edge_factor)
        weights_y = torch.exp(-image_dy * self.edge_factor)
        
        # Ограничиваем веса для устойчивости к квантизации
        weights_x = torch.clamp(weights_x, 0.1, 1.0)
        weights_y = torch.clamp(weights_y, 0.1, 1.0)
        
        # Финальная потеря с пространственной нормализацией для баланса
        smoothness_x = (depth_dx * weights_x).mean() / (weights_x.mean() + 1e-7)
        smoothness_y = (depth_dy * weights_y).mean() / (weights_y.mean() + 1e-7)
        
        return (smoothness_x + smoothness_y) * self.weight

class SSIM(nn.Module):
    """
    Улучшенное структурное сходство (SSIM) с механизмами стабилизации
    для оценки визуального качества карты глубины и устойчивости к квантизации.
    """
    def __init__(self, window_size=7, size_average=True, C1=0.01, C2=0.03):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        # Увеличенные константы для стабильности при квантизации
        self.C1 = C1**2
        self.C2 = C2**2
        
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def _gaussian(self, window_size, sigma):
        # Создаем тензор с координатами
        coords = torch.arange(window_size, dtype=torch.float)
        # Вычисляем центр окна
        center = window_size // 2
        # Вычисляем квадрат расстояния от центра (преобразовано в тензор)
        sq_diff = (coords - center) ** 2
        # Вычисляем гауссову функцию на тензоре
        gauss = torch.exp(-sq_diff / (2 * sigma ** 2))
        # Нормализуем
        return gauss / gauss.sum()
    
    def forward(self, pred, target):
        # Перемещаем окно на то же устройство, что и входные данные
        if not hasattr(self, 'window') or self.window.device != pred.device:
            self.window = self._create_window(self.window_size, self.channel).to(pred.device)
        
        # Cтабилизация входных данных для квантизации
        pred = torch.clamp(pred, 0.001, 0.999)
        target = torch.clamp(target, 0.001, 0.999)
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # Дополнительная стабилизация числителя и знаменателя
        numer = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denom = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        # Cтабилизация для избежания деления на очень маленькие числа
        ssim_map = numer / (denom + 1e-8)
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3])

class EdgeAwareLoss(nn.Module):
    """
    Специальная потеря для сохранения границ объектов разной глубины,
    критично для качественной квантизации.
    """
    def __init__(self, weight=0.2, sobel_size=3):
        super(EdgeAwareLoss, self).__init__()
        self.weight = weight
        self.sobel_kernel_x = torch.FloatTensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).unsqueeze(0).unsqueeze(0)
        
        self.sobel_kernel_y = torch.FloatTensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).unsqueeze(0).unsqueeze(0)
        
    def forward(self, pred, target):
        device = pred.device
        
        if not hasattr(self, 'sobel_kernel_x_device') or self.sobel_kernel_x_device.device != device:
            self.sobel_kernel_x_device = self.sobel_kernel_x.to(device)
            self.sobel_kernel_y_device = self.sobel_kernel_y.to(device)
        
        # Конвертация в edge maps с использованием оператора Собеля
        edge_x_pred = F.conv2d(pred, self.sobel_kernel_x_device, padding=1)
        edge_y_pred = F.conv2d(pred, self.sobel_kernel_y_device, padding=1)
        
        edge_x_target = F.conv2d(target, self.sobel_kernel_x_device, padding=1)
        edge_y_target = F.conv2d(target, self.sobel_kernel_y_device, padding=1)
        
        # Magnitude градиентов
        edge_pred = torch.sqrt(edge_x_pred**2 + edge_y_pred**2 + 1e-6)
        edge_target = torch.sqrt(edge_x_target**2 + edge_y_target**2 + 1e-6)
        
        # Подчеркиваем потери на ярко выраженных границах
        edge_mask = edge_target > 0.05
        
        # Если есть значимые края, расчитываем потери усиленные на краях
        if torch.any(edge_mask):
            # L1 ошибка на краях
            edge_error = torch.abs(edge_pred[edge_mask] - edge_target[edge_mask]).mean()
            
            # Учитываем также направление градиента для более точного сохранения границ
            if torch.any(edge_x_target != 0) and torch.any(edge_y_target != 0):
                # Нормализованные векторы направлений
                norm_x_pred = edge_x_pred / (edge_pred + 1e-7)
                norm_y_pred = edge_y_pred / (edge_pred + 1e-7)
                
                norm_x_target = edge_x_target / (edge_target + 1e-7)
                norm_y_target = edge_y_target / (edge_target + 1e-7)
                
                # Скалярное произведение для угла между векторами
                dot_product = norm_x_pred * norm_x_target + norm_y_pred * norm_y_target
                angle_error = 1 - torch.abs(dot_product[edge_mask]).mean()
                
                # Комбинированная edge-aware потеря
                return (edge_error + angle_error * 0.5) * self.weight
            
            return edge_error * self.weight
        
        # Если нет значимых краев, возвращаем минимальную потерю
        return torch.tensor(0.0, device=device)

class RobustLoss(nn.Module):
    """
    Комбинация функций потерь для улучшения стабильности квантизированных моделей
    и качества предсказания глубины. Оптимизирована по весам для наилучшего соотношения
    качества и устойчивости к квантизации.
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1, edge_weight=0.15):
        super(RobustLoss, self).__init__()
        self.alpha = alpha  # вес для L1
        self.beta = beta    # вес для BerHu
        self.gamma = gamma  # вес для Scale-Invariant
        self.delta = delta  # вес для SSIM
        self.edge_weight = edge_weight  # вес для Edge-Aware потери
        
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.berhu_loss = BerHuLoss(threshold=0.2)
        self.scale_invariant = ScaleInvariantLoss()
        self.ssim_loss = SSIM(window_size=7)
        self.edge_loss = EdgeAwareLoss(weight=1.0)
        
        # Создадим директорию для визуализаций
        self.vis_dir = None
        
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Создает визуализацию для анализа работы функции потерь"""
        if self.vis_dir is None:
            self.vis_dir = output_dir
            os.makedirs(self.vis_dir, exist_ok=True)

        # Берем первый элемент из батча для визуализации
        if isinstance(pred, list):
            # Берем первый элемент из списка предсказаний (для multi-scale выходов)
            pred_item = pred[0][0, 0].detach().cpu().numpy()  # Исправить здесь - берем [0, 0] вместо [0]
        else:
            pred_item = pred[0, 0].detach().cpu().numpy()

        target_item = target[0, 0].detach().cpu().numpy()
        rgb_item = rgb[0].detach().cpu().permute(1, 2, 0).numpy()

        # Проверка формы и сжатие при необходимости
        if len(pred_item.shape) > 2:
            pred_item = pred_item.squeeze()  # Удаляем лишние размерности
        if len(target_item.shape) > 2:
            target_item = target_item.squeeze()

        # Денормализуем RGB
        rgb_item = rgb_item * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_item = np.clip(rgb_item, 0, 1)

        # Создаем карту ошибок
        error_map = np.abs(pred_item - target_item)

        # Визуализируем Sobel-фильтры для анализа краев
        device = target.device
        dtype = target.dtype  # Получаем тип данных из входных тензоров

        # Для безопасности переводим на CPU и используем float32 для операций
        # Используем правильный формат для pred при применении фильтров
        if isinstance(pred, list):
            pred_tensor = pred[0]
        else:
            pred_tensor = pred

        pred_cpu = pred_tensor[0:1].detach().cpu().float()
        target_cpu = target[0:1].detach().cpu().float()

        # Создаем Sobel-фильтры с правильным типом данных
        sobel_x_cpu = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float()
        sobel_y_cpu = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float()

        edge_x_pred = F.conv2d(pred_cpu, sobel_x_cpu, padding=1)
        edge_y_pred = F.conv2d(pred_cpu, sobel_y_cpu, padding=1)
        edge_pred = torch.sqrt(edge_x_pred**2 + edge_y_pred**2).numpy()[0, 0]

        edge_x_target = F.conv2d(target_cpu, sobel_x_cpu, padding=1)
        edge_y_target = F.conv2d(target_cpu, sobel_y_cpu, padding=1)
        edge_target = torch.sqrt(edge_x_target**2 + edge_y_target**2).numpy()[0, 0]

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
        plt.close()
        
    def forward(self, pred, target, rgb=None):
        # Для multi-scale выходов используем только основной выход для упрощения
        if isinstance(pred, list):
            pred_solo = pred[0]
        else:
            pred_solo = pred
        
        # Стабилизация входных данных
        pred_solo = torch.clamp(pred_solo, 1e-3, 1.0 - 1e-3)
        target = torch.clamp(target, 1e-3, 1.0 - 1e-3)
        
        # Вычисляем базовые потери
        l1_loss = self.l1_loss(pred_solo, target)
        berhu_loss = self.berhu_loss(pred_solo, target)
        si_loss = self.scale_invariant(pred_solo, target)
        
        # SSIM потеря, если размер входа не слишком большой
        if pred_solo.shape[2] <= 384:  # Оптимизация для больших размеров
            ssim_loss = self.ssim_loss(pred_solo, target)
        else:
            ssim_loss = torch.tensor(0.0, device=pred_solo.device)
        
        # Edge-aware потери если доступно RGB изображение
        if rgb is not None:
            edge_aware = self.edge_loss(pred_solo, target)
            return (self.alpha * l1_loss + 
                    self.beta * berhu_loss + 
                    self.gamma * si_loss + 
                    self.delta * ssim_loss +
                    self.edge_weight * edge_aware)
        else:
            return (self.alpha * l1_loss + 
                    self.beta * berhu_loss + 
                    self.gamma * si_loss + 
                    self.delta * ssim_loss)

class DepthWithSmoothnessLoss(nn.Module):
    """
    Оптимизированная версия комбинированной функции потерь
    для улучшения плавности карты глубины и устранения артефактов
    с механизмами защиты от квантизации.
    """
    def __init__(self, base_weight=0.7, smoothness_weight=0.15, gradient_weight=0.15):
        super(DepthWithSmoothnessLoss, self).__init__()
        self.base_weight = base_weight
        self.smoothness_weight = smoothness_weight
        self.gradient_weight = gradient_weight
        
        # Используем более стабильные функции потерь
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(window_size=7, C1=0.01, C2=0.03)  # Увеличенные константы стабильности
        self.smoothness_loss = SmoothnessLoss(weight=1.0, edge_factor=10.0)
        self.gradient_loss = GradientLoss(weight=1.0, edge_threshold=0.05)
        
        # Директория для визуализаций
        self.vis_dir = None
        
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Создает визуализацию для анализа работы функции потерь"""
        if self.vis_dir is None:
            self.vis_dir = output_dir
            os.makedirs(self.vis_dir, exist_ok=True)
    
        # Берем первый элемент из батча для визуализации
        if isinstance(pred, list):
            # Берем первый элемент из списка предсказаний (для multi-scale выходов)
            pred_item = pred[0][0, 0].detach().cpu().numpy()  # Исправить здесь - берем [0, 0] вместо [0]
        else:
            pred_item = pred[0, 0].detach().cpu().numpy()
            
        target_item = target[0, 0].detach().cpu().numpy()
        rgb_item = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
        
        # Проверка формы и сжатие при необходимости
        if len(pred_item.shape) > 2:
            pred_item = pred_item.squeeze()  # Удаляем лишние размерности
        if len(target_item.shape) > 2:
            target_item = target_item.squeeze()
            
        # Денормализуем RGB
        rgb_item = rgb_item * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_item = np.clip(rgb_item, 0, 1)
        
        # Создаем карту ошибок
        error_map = np.abs(pred_item - target_item)
        
        # Визуализируем Sobel-фильтры для анализа краев
        device = target.device
        dtype = target.dtype  # Получаем тип данных из входных тензоров
        
        # Для безопасности переводим на CPU и используем float32 для операций
        # Используем правильный формат для pred при применении фильтров
        if isinstance(pred, list):
            pred_tensor = pred[0]
        else:
            pred_tensor = pred
        
        pred_cpu = pred_tensor[0:1].detach().cpu().float()
        target_cpu = target[0:1].detach().cpu().float()
        
        # Создаем Sobel-фильтры с правильным типом данных
        sobel_x_cpu = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float()
        sobel_y_cpu = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float()
        
        edge_x_pred = F.conv2d(pred_cpu, sobel_x_cpu, padding=1)
        edge_y_pred = F.conv2d(pred_cpu, sobel_y_cpu, padding=1)
        edge_pred = torch.sqrt(edge_x_pred**2 + edge_y_pred**2).numpy()[0, 0]
        
        edge_x_target = F.conv2d(target_cpu, sobel_x_cpu, padding=1)
        edge_y_target = F.conv2d(target_cpu, sobel_y_cpu, padding=1)
        edge_target = torch.sqrt(edge_x_target**2 + edge_y_target**2).numpy()[0, 0]
    
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
        plt.close()
        
    def forward(self, pred, target, rgb=None):
        # Для multi-scale выходов используем только основной выход для упрощения
        if isinstance(pred, list):
            pred_solo = pred[0]
        else:
            pred_solo = pred
            
        # Стабилизация входных данных для квантизации
        pred_solo = torch.clamp(pred_solo, 0.001, 0.999)
        target = torch.clamp(target, 0.001, 0.999)
        
        # Основная потеря - комбинация L1 и SSIM
        l1_loss = self.l1_loss(pred_solo, target)
        
        # Вычисляем SSIM только если depth_pred маленького размера или если используется смешанная точность
        use_simplified = pred_solo.shape[2] > 384  # Для очень больших изображений используем упрощенную функцию
        
        if use_simplified:
            # Упрощенный вариант - только L1 для ускорения
            base_loss = l1_loss
        else:
            # Полный вариант с SSIM
            ssim_loss = self.ssim_loss(pred_solo, target)
            base_loss = 0.6 * l1_loss + 0.4 * ssim_loss  # Увеличенный вес SSIM
        
        # Потеря сглаживания с учетом границ и устойчивости к квантизации
        if rgb is not None and not use_simplified:
            smooth_loss = self.smoothness_loss(pred_solo, rgb)
            gradient_loss = self.gradient_loss(pred_solo, target)
        else:
            smooth_loss = 0.0
            gradient_loss = 0.0
            
        # Формируем итоговую потерю с учетом весов
        total_loss = self.base_weight * base_loss
        if smooth_loss > 0:
            total_loss += self.smoothness_weight * smooth_loss
        if gradient_loss > 0:
            total_loss += self.gradient_weight * gradient_loss
            
        return total_loss

class MultiScaleLoss(nn.Module):
    """
    Улучшенная loss function для работы с multi-scale выходами модели и
    повышения устойчивости к квантизации.
    
    Специально адаптирована для B_MobileDepth модели.
    """
    def __init__(self, base_criterion=None, weights=None):
        super(MultiScaleLoss, self).__init__()
        self.base_criterion = base_criterion or RobustLoss(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
        self.weights = weights or [0.6, 0.2, 0.1, 0.1]  # Веса для разных масштабов
        
        # Директория для визуализаций
        self.vis_dir = None
        
    def _visualize_batch(self, pred, target, rgb, batch_idx, epoch, output_dir="loss_visualizations"):
        """Делегирует визуализацию базовому критерию"""
        if hasattr(self.base_criterion, '_visualize_batch'):
            self.base_criterion._visualize_batch(pred, target, rgb, batch_idx, epoch, output_dir)
    
    def forward(self, pred, target, rgb=None):
        # Если prediction is a list (multi-scale), compute weighted loss
        if isinstance(pred, list):
            total_loss = 0
            
            # Ensure we have weights for each output
            weights = self.weights
            if len(weights) != len(pred):
                weights = [1.0/len(pred)] * len(pred)
            
            # Normalize weights to sum to 1
            sum_weights = sum(weights)
            weights = [w/sum_weights for w in weights]
            
            # Calculate loss for each scale with increasing weight for finer scales
            for i, pred_i in enumerate(pred):
                # Интерполируем предсказание, если разные размеры
                if pred_i.shape[2:] != target.shape[2:]:
                    pred_i = F.interpolate(
                        pred_i, 
                        size=target.shape[2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Передаем RGB изображение, если оно доступно
                if rgb is not None:
                    loss = self.base_criterion(pred_i, target, rgb)
                else:
                    loss = self.base_criterion(pred_i, target)
                    
                total_loss += weights[i] * loss
                
            return total_loss
        else:
            # Single output, just use base criterion
            if rgb is not None:
                return self.base_criterion(pred, target, rgb)
            else:
                return self.base_criterion(pred, target)
