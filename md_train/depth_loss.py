import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
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
    Потеря градиента для улучшения согласованности и плавности карты глубины,
    особенно эффективна для борьбы с артефактами и "плоскими" регионами.
    """
    def __init__(self, weight=1.0):
        super(GradientLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        # Градиенты по x и y для предсказания
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        # Градиенты по x и y для ground truth
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # Ошибка градиентов
        grad_diff_x = torch.abs(pred_dx - target_dx)
        grad_diff_y = torch.abs(pred_dy - target_dy)
        
        return (grad_diff_x.mean() + grad_diff_y.mean()) * self.weight

class SmoothnessLoss(nn.Module):
    """
    Потеря сглаживания с учетом границ в изображении.
    Уменьшает резкие изменения в глубине, за исключением мест, где есть границы на изображении.
    """
    def __init__(self, weight=0.1):
        super(SmoothnessLoss, self).__init__()
        self.weight = weight
        
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
        
        # Уменьшаем штраф, если есть края на изображении
        weights_x = torch.exp(-image_dx * 10)
        weights_y = torch.exp(-image_dy * 10)
        
        # Финальная потеря
        smoothness_x = depth_dx * weights_x
        smoothness_y = depth_dy * weights_y
        
        return (smoothness_x.mean() + smoothness_y.mean()) * self.weight

class SSIM(nn.Module):
    """
    Структурное сходство (SSIM) для оценки визуального качества карты глубины
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
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
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3])

class RobustLoss(nn.Module):
    """
    Комбинация функций потерь для улучшения стабильности квантизированных моделей
    и качества предсказания глубины
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super(RobustLoss, self).__init__()
        self.alpha = alpha  # вес для L1
        self.beta = beta    # вес для BerHu
        self.gamma = gamma  # вес для Scale-Invariant
        self.delta = delta  # вес для SSIM
        
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.berhu_loss = BerHuLoss(threshold=0.2)
        self.scale_invariant = ScaleInvariantLoss()
        self.ssim_loss = SSIM()
        
    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        berhu_loss = self.berhu_loss(pred, target)
        si_loss = self.scale_invariant(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        
        return (self.alpha * l1_loss + 
                self.beta * berhu_loss + 
                self.gamma * si_loss + 
                self.delta * ssim_loss)

class DepthWithSmoothnessLoss(nn.Module):
    """
    Оптимизированная версия комбинированной функции потерь
    для улучшения плавности карты глубины и устранения артефактов
    """
    def __init__(self, base_weight=0.85, smoothness_weight=0.15):
        super(DepthWithSmoothnessLoss, self).__init__()
        self.base_weight = base_weight
        self.smoothness_weight = smoothness_weight
        
        # Используем более быстрые функции потерь
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(window_size=7)  # Меньшее окно для ускорения
        self.smoothness_loss = SmoothnessLoss(weight=1.0)
        
    def forward(self, pred, target, rgb=None):
        # Основная потеря - комбинация L1 и SSIM
        l1_loss = self.l1_loss(pred, target)
        
        # Вычисляем SSIM только если depth_pred маленького размера или если используется смешанная точность
        use_simplified = pred.shape[2] > 128  # Для больших изображений используем упрощенную функцию
        
        if use_simplified:
            # Упрощенный вариант - только L1 для ускорения
            base_loss = l1_loss
        else:
            # Полный вариант с SSIM
            ssim_loss = self.ssim_loss(pred, target)
            base_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        # Потеря сглаживания с учетом границ, если есть RGB изображение
        if rgb is not None and not use_simplified:
            smooth_loss = self.smoothness_loss(pred, rgb)
        else:
            smooth_loss = 0.0
            
        # Формируем итоговую потерю с учетом весов
        total_loss = self.base_weight * base_loss
        if smooth_loss > 0:
            total_loss += self.smoothness_weight * smooth_loss
            
        return total_loss

class MultiScaleLoss(nn.Module):
    """Loss function that can handle multi-scale outputs from the model"""
    def __init__(self, base_criterion=None, weights=None):
        super(MultiScaleLoss, self).__init__()
        self.base_criterion = base_criterion or DepthWithSmoothnessLoss()  # Используем нашу улучшенную функцию потерь
        self.weights = weights or [0.7, 0.2, 0.1]  # Default weights for 3 scales
    
    def forward(self, pred, target, rgb=None):
        # Если prediction is a list (multi-scale), compute weighted loss
        if isinstance(pred, list):
            total_loss = 0
            
            # Ensure we have weights for each output
            weights = self.weights
            if len(weights) != len(pred):
                weights = [1.0/len(pred)] * len(pred)
            
            # Calculate loss for each scale
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
