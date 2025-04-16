import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """Оптимизированный для квантизации блок свертки с BatchNorm и ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            groups=groups, bias=False
        )
        # Используем большее значение eps для повышения стабильности при квантизации
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-3)
        # Используем ReLU6 для лучшей квантизации с ограниченным диапазоном активации
        self.relu = nn.ReLU(inplace=True)  # Замена ReLU6 на ReLU

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable свертка, оптимизированная для квантизации"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = ConvBNReLU(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels
        )
        self.pointwise = ConvBNReLU(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EnhancedQuantResidualBlock(nn.Module):
    """Улучшенный residual блок с учетом квантизации"""
    def __init__(self, channels, expansion=1):
        super(EnhancedQuantResidualBlock, self).__init__()
        expanded_channels = channels * expansion
        
        # Первая конволюция с расширением каналов
        self.conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)  # ReLU вместо ReLU6
        
        # Вторая конволюция с группировкой для лучшей квантизации
        self.conv2 = nn.Conv2d(
            expanded_channels, expanded_channels, 
            kernel_size=3, padding=1, groups=4, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expanded_channels, eps=1e-3)
        
        # Проекция для residual если нужно
        self.shortcut = nn.Identity()
        if expansion > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels, eps=1e-3)
            )
        
        self.relu2 = nn.ReLU(inplace=True)  # ReLU вместо ReLU6
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu2(out)
        
        return out

class QuantizationFriendlyDetailModule(nn.Module):
    """Enhanced detail preservation module with quantization robustness"""
    def __init__(self, channels):
        super(QuantizationFriendlyDetailModule, self).__init__()
        # Use smaller convolution groups for better quantization
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)  # ReLU
        
        # 1x1 convolution for channel mixing with controlled range
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)
        
        # Final detail enhancement with residual connection
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels, eps=1e-3)
        self.relu2 = nn.ReLU(inplace=True)  # ReLU
        
        # Final detail enhancement with residual connection # Edge detection branch
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.edge_bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.edge_weight = nn.Parameter(torch.ones(1) * 0.1)  # Learnable weight for edge features
        
    def forward(self, x):
        residual = x
        
        # Main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Edge detection branch (with controlled contribution)
        edge = self.edge_conv(x)
        edge = self.edge_bn(edge)
        edge = torch.sigmoid(edge)  # Sigmoid activation for edge features # Keep in [0,1] range
        
        # Combine main features with edge features
        out = out + (edge * self.edge_weight)
        out = out + residual      # Add residual connection
        out = self.relu2(out)     # Final bounded activation
        
        return out

class EdgeAwareFeatureEnhancement(nn.Module):
    """Улучшенный модуль обработки границ для повышения детализации границ глубины"""
    def __init__(self, channels):
        super(EdgeAwareFeatureEnhancement, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        # Используем depthwise для лучшей квантизации
        self.conv2 = ConvBNReLU(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # Edge-aware обработка с 1x1 сверткой для лучшей квантизации
        self.edge_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                                   groups=channels, bias=False)
        self.edge_bn1 = nn.BatchNorm2d(channels, eps=1e-3)
        self.edge_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.edge_bn2 = nn.BatchNorm2d(channels, eps=1e-3)
        
        # Финальная свертка с residual соединением
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)  # ReLU
        
    def forward(self, x):
        residual = x
        # Основная ветвь
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Edge-aware ветвь
        edge = self.edge_conv1(x)
        edge = self.edge_bn1(edge)
        edge = F.relu(edge)  # ReLU
        edge = self.edge_conv2(edge)
        edge = self.edge_bn2(edge)
        
        # Комбинируем признаки с edge информацией
        x = x + edge * 0.1
        
        # Финальная обработка
        x = self.conv3(x)
        x = self.bn(x)
        x = x + residual
        x = self.relu(x)
        
        return x

class QuantFriendlyAttention(nn.Module):
    """Улучшенный квантизационно-дружественный модуль внимания"""
    def __init__(self, channels):
        super(QuantFriendlyAttention, self).__init__()
        # Уменьшаем размерность для эффективности
        self.conv1 = ConvBNReLU(channels, channels // 2, kernel_size=1)
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(channels // 2, 1, kernel_size=7, padding=3, bias=False)
        self.bn_spatial = nn.BatchNorm2d(1, eps=1e-3)
        
        # Channel attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_channel = nn.Conv2d(channels // 2, channels // 8, kernel_size=1, bias=False)
        self.bn_channel = nn.BatchNorm2d(channels // 8, eps=1e-3)
        self.conv_channel2 = nn.Conv2d(channels // 8, channels, kernel_size=1, bias=False)
        self.bn_channel2 = nn.BatchNorm2d(channels, eps=1e-3)
        
    def forward(self, x):
        features = self.conv1(x)
        
        # Spatial attention
        spatial_att = self.conv_spatial(features)
        spatial_att = self.bn_spatial(spatial_att)
        spatial_att = torch.sigmoid(spatial_att)
        
        # Channel attention
        channel_att = self.pool(features)
        channel_att = self.conv_channel(channel_att)
        channel_att = self.bn_channel(channel_att)
        channel_att = F.relu(channel_att)  # ReLU # Ограниченная активация
        channel_att = self.conv_channel2(channel_att)
        channel_att = self.bn_channel2(channel_att)
        channel_att = torch.sigmoid(channel_att)
        
        # Комбинированное внимание
        x = x * spatial_att * channel_att
        
        return x

class UpBlockWithAttention(nn.Module):
    """Улучшенный блок апсемплинга с вниманием и лучшей поддержкой квантизации"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlockWithAttention, self).__init__()
        self.scale_factor = scale_factor
        
        # Предварительная свертка для уменьшения каналов
        self.conv_before = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        
        # Residual блок после upsampling
        self.conv_after = EnhancedQuantResidualBlock(out_channels)
        
        # Модуль улучшения деталей
        self.detail_enhance = QuantizationFriendlyDetailModule(out_channels)
        
        # Attention для акцентирования важных областей
        self.attention = QuantFriendlyAttention(out_channels)
        
    def forward(self, x):
        # Предварительная свертка для уменьшения размерности
        x_reduced = self.conv_before(x)
        
        # Upsampling с фиксированным коэффициентом - сначала ближайший сосед, затем сглаживание
        x_up = F.interpolate(
            x_reduced, 
            scale_factor=self.scale_factor, 
            mode='nearest'  # менее подвержен квантизационным артефактам
        )
        
        # Сглаживающая свертка для устранения артефактов nearest upsampling
        x_up = self.conv_after(x_up)
        
        # Улучшение деталей
        out = self.detail_enhance(x_up)
        
        # Attention для фокусирования на важных областях
        out = self.attention(out)
        
        return out

class FeatureFusion(nn.Module):
    """Улучшенный модуль слияния признаков для более эффективного объединения skip-connections"""
    def __init__(self, main_channels, skip_channels, out_channels):
        super(FeatureFusion, self).__init__()
        # Свертки для приведения каналов к единой размерности
        self.conv_main = ConvBNReLU(main_channels, out_channels, kernel_size=1)
        self.conv_skip = ConvBNReLU(skip_channels, out_channels, kernel_size=1)
        
        # Adaptive fusion с вниманием
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.Sigmoid()
        )
        
        # Финальная свертка для интеграции признаков
        self.fusion = ConvBNReLU(out_channels*2, out_channels, kernel_size=3, padding=1)
        
    def forward(self, main, skip):
        # Приведение размерностей
        main = self.conv_main(main)
        skip = self.conv_skip(skip)
        
        # Расчет весов для адаптивного слияния
        concat = torch.cat([main, skip], dim=1)
        gate = self.fusion_gate(concat)
        
        # Взвешенное слияние
        fused_features = main * gate + skip * (1 - gate)
        
        # Конкатенация и финальная интеграция
        fused = torch.cat([fused_features, skip], dim=1)
        out = self.fusion(fused)
        
        return out

class B_MobileDepth(nn.Module):
    """
    Улучшенная MobileDepth модель с multi-scale выходами, углублёнными skip-connections 
    и расширенным bottleneck для лучшей детализации и устойчивости к квантизации.
    """
    def __init__(self, input_size=(320, 320)):
        super(B_MobileDepth, self).__init__()
        self.input_size = input_size
        
        # Encoder с увеличенным количеством фильтров и residual блоков
        self.enc_conv1 = ConvBNReLU(3, 48, kernel_size=3, stride=2, padding=1)       # 160x160
        self.enc_res1 = EnhancedQuantResidualBlock(48)
        
        self.enc_ds_conv1 = DepthwiseSeparableConv(48, 96, stride=1)                 # 160x160
        self.enc_res2 = EnhancedQuantResidualBlock(96)
        
        self.enc_ds_conv2 = DepthwiseSeparableConv(96, 128, stride=2)                # 80x80
        self.enc_ds_conv3 = DepthwiseSeparableConv(128, 128, stride=1)               # 80x80
        self.enc_res3 = EnhancedQuantResidualBlock(128)
        
        self.enc_ds_conv4 = DepthwiseSeparableConv(128, 256, stride=2)               # 40x40
        self.enc_ds_conv5 = DepthwiseSeparableConv(256, 256, stride=1)               # 40x40
        self.enc_res4 = EnhancedQuantResidualBlock(256)
        
        self.enc_ds_conv6 = DepthwiseSeparableConv(256, 512, stride=2)               # 20x20
        # Расширенный bottleneck с дополнительным residual блоком
        self.enc_ds_conv7 = DepthwiseSeparableConv(512, 768, stride=1)               # 20x20
        self.enc_res5 = EnhancedQuantResidualBlock(768)
        
        # Улучшенные блоки внимания для skip-connections
        self.att_skip1 = QuantFriendlyAttention(96)
        self.att_skip2 = QuantFriendlyAttention(128)
        self.att_skip3 = QuantFriendlyAttention(256)
        
        # Улучшенные блоки обработки признаков на skip-connections
        self.edge_aware1 = EdgeAwareFeatureEnhancement(96)
        self.edge_aware2 = EdgeAwareFeatureEnhancement(128)
        self.edge_aware3 = EdgeAwareFeatureEnhancement(256)
        
        # Decoder с улучшенным слиянием признаков и attention модулями
        self.up1 = UpBlockWithAttention(768, 256, scale_factor=2)                    # 20x20 -> 40x40
        self.fusion1 = FeatureFusion(256, 256, 256)
        
        self.up2 = UpBlockWithAttention(256, 128, scale_factor=2)                    # 40x40 -> 80x80
        self.fusion2 = FeatureFusion(128, 128, 128)
        
        self.up3 = UpBlockWithAttention(128, 96, scale_factor=2)                     # 80x80 -> 160x160
        self.fusion3 = FeatureFusion(96, 96, 96)
        
        self.up4 = UpBlockWithAttention(96, 48, scale_factor=2)                      # 160x160 -> 320x320
        
        # Multi-scale выходы для улучшения обучения на разных масштабах
        self.out_conv1 = nn.Sequential(
            ConvBNReLU(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.out_conv2 = nn.Sequential(
            ConvBNReLU(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.out_conv3 = nn.Sequential(
            ConvBNReLU(96, 48, kernel_size=3, padding=1),
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Модуль улучшения деталей в финальном выходе
        self.detail_enhance = QuantizationFriendlyDetailModule(48)
        
        # Финальные свертки
        self.final_conv1 = ConvBNReLU(48, 24, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(24, 1, kernel_size=3, padding=1)
        
        # Инициализация весов для лучшей квантизации
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Используем инициализацию Каиминга для лучшей квантизации
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Ограничиваем начальные веса
                with torch.no_grad():
                    m.weight.data.clamp_(-1.0, 1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Проверка размеров входных данных
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        
        # Encoder
        enc1 = self.enc_conv1(x)          # 48 x 160 x 160
        enc1 = self.enc_res1(enc1)        # 48 x 160 x 160
        
        skip1 = self.enc_ds_conv1(enc1)   # 96 x 160 x 160
        skip1 = self.enc_res2(skip1)      # 96 x 160 x 160
        
        enc2 = self.enc_ds_conv2(skip1)   # 128 x 80 x 80
        skip2 = self.enc_ds_conv3(enc2)   # 128 x 80 x 80
        skip2 = self.enc_res3(skip2)      # 128 x 80 x 80
        
        enc3 = self.enc_ds_conv4(skip2)   # 256 x 40 x 40
        skip3 = self.enc_ds_conv5(enc3)   # 256 x 40 x 40
        skip3 = self.enc_res4(skip3)      # 256 x 40 x 40
        
        enc4 = self.enc_ds_conv6(skip3)   # 512 x 20 x 20
        enc4 = self.enc_ds_conv7(enc4)    # 768 x 20 x 20
        enc4 = self.enc_res5(enc4)        # 768 x 20 x 20
        
        # Улучшение признаков на skip-connections
        skip1 = self.edge_aware1(skip1)
        skip2 = self.edge_aware2(skip2)
        skip3 = self.edge_aware3(skip3)
        
        # Attention modules
        skip1_att = self.att_skip1(skip1)
        skip2_att = self.att_skip2(skip2)
        skip3_att = self.att_skip3(skip3)
        
        # Decoder с улучшенным слиянием признаков
        d1 = self.up1(enc4)                             # 256 x 40 x 40
        d1 = self.fusion1(d1, skip3_att)                # 256 x 40 x 40
        out_scale1 = self.out_conv1(d1)                 # 1 x 40 x 40
        
        d2 = self.up2(d1)                               # 128 x 80 x 80
        d2 = self.fusion2(d2, skip2_att)                # 128 x 80 x 80
        out_scale2 = self.out_conv2(d2)                 # 1 x 80 x 80
        
        d3 = self.up3(d2)                               # 96 x 160 x 160
        d3 = self.fusion3(d3, skip1_att)                # 96 x 160 x 160
        out_scale3 = self.out_conv3(d3)                 # 1 x 160 x 160
        
        d4 = self.up4(d3)                               # 48 x 320 x 320
        
        # Улучшение деталей в финальном выходе
        d4 = self.detail_enhance(d4)                    # 48 x 320 x 320
        
        # Финальные свертки
        out = self.final_conv1(d4)                      # 24 x 320 x 320
        depth = self.final_conv2(out)                   # 1 x 320 x 320
        
        # Используем hard_sigmoid для плавной активации вместо обычного sigmoid для лучшей квантизации
        depth = torch.clamp(depth * 0.16667 + 0.5, 0, 1)
        
        # Resize отмасштабированные выходы под размер основного выхода для multi-scale loss
        if self.training:
            out_scale1 = F.interpolate(out_scale1, size=depth.shape[2:], mode='bilinear', align_corners=False)
            out_scale2 = F.interpolate(out_scale2, size=depth.shape[2:], mode='bilinear', align_corners=False)
            out_scale3 = F.interpolate(out_scale3, size=depth.shape[2:], mode='bilinear', align_corners=False)
            return [depth, out_scale1, out_scale2, out_scale3]
        
        return depth

if __name__ == "__main__":
    # Проверка модели
    model = B_MobileDepth(input_size=(320, 320))
    dummy_input = torch.randn(1, 3, 320, 320)
    
    # Печать информации о модели
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Тест прямого прохода
    with torch.no_grad():
        outputs = model(dummy_input)
        if isinstance(outputs, list):
            print(f"Multi-scale outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"  Scale {i}, shape: {out.shape}")
        else:
            output = outputs
            print(f"Output shape: {output.shape}")
            print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
            print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
