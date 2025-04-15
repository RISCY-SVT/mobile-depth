import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """Оптимизированный для квантизации блок свертки с BatchNorm и ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        # Используем большее значение eps для повышения стабильности при квантизации
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-3)
        # Используем ReLU вместо ReLU6, так как ReLU6 может создавать проблемы при квантизации
        self.relu = nn.ReLU(inplace=True)

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

class DetailEnhancementModule(nn.Module):
    """Модуль улучшения деталей для повышения детализации карты глубины"""
    def __init__(self, channels):
        super(DetailEnhancementModule, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBNReLU(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = x + residual
        return self.relu(x)

class EdgeAwareFeatureEnhancement(nn.Module):
    """Модуль обработки границ для улучшения детектирования переходов глубины"""
    def __init__(self, channels):
        super(EdgeAwareFeatureEnhancement, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBNReLU(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = x + residual
        return self.relu(x)

class QuantFriendlyAttention(nn.Module):
    """Улучшенный квантизационно-дружественный модуль внимания"""
    def __init__(self, channels):
        super(QuantFriendlyAttention, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 2, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        # Используем ReLU для активации вместо hardtanh для лучшей квантизации
        
    def forward(self, x):
        features = self.conv1(x)
        attention = self.conv2(features)
        attention = self.bn(attention)
        attention = torch.sigmoid(attention)  # Более плавная активация
        
        # Нормализуем внимание для избежания артефактов квантизации
        return x * attention

class ResidualBlock(nn.Module):
    """Квантизационно-дружественный остаточный блок"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        # Добавляем остаточное соединение и активацию после сложения
        # для лучшей квантизации
        out += residual
        out = self.relu(out)
        return out

class UpBlock(nn.Module):
    """Улучшенный блок апсемплинга с лучшей поддержкой квантизации"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv_before = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.conv_after = ResidualBlock(out_channels)
        self.detail_enhance = DetailEnhancementModule(out_channels)
        
    def forward(self, x):
        # Pre-upsampling convolution для уменьшения размера тензора до апсемплинга
        x_reduced = self.conv_before(x)
        
        # Upsampling с фиксированным коэффициентом - лучше для квантизации
        x_up = F.interpolate(
            x_reduced, 
            scale_factor=self.scale_factor, 
            mode='nearest'  # используем nearest вместо bilinear для лучшей квантизации
        )
        
        # Применение операции после апсемплинга
        out = self.conv_after(x_up)
        
        # Улучшение деталей
        out = self.detail_enhance(out)
        
        return out

class FeatureFusion(nn.Module):
    """Модуль слияния признаков с вниманием для лучшего объединения skip-connections"""
    def __init__(self, main_channels, skip_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv_main = ConvBNReLU(main_channels, out_channels, kernel_size=1)
        self.conv_skip = ConvBNReLU(skip_channels, out_channels, kernel_size=1)
        self.fusion = ConvBNReLU(out_channels*2, out_channels, kernel_size=3, padding=1)
        
    def forward(self, main, skip):
        main = self.conv_main(main)
        skip = self.conv_skip(skip)
        
        # Конкатенация вместо суммирования для лучшей квантизации
        fused = torch.cat([main, skip], dim=1)
        out = self.fusion(fused)
        
        return out

class ImprovedQuantizationFriendlyMobileDepth(nn.Module):
    """
    Улучшенная MobileDepth модель для лучшей детализации и устойчивости к квантизации
    на NPU AX3386 (LicheePi4A)
    """
    def __init__(self, input_size=(256, 256)):
        super(ImprovedQuantizationFriendlyMobileDepth, self).__init__()
        self.input_size = input_size
        
        # Encoder с увеличенным количеством фильтров для лучшей детализации
        self.enc_conv1 = ConvBNReLU(3, 48, kernel_size=3, stride=2, padding=1)       # 128x128
        self.enc_ds_conv1 = DepthwiseSeparableConv(48, 96, stride=1)                 # 128x128
        self.enc_ds_conv2 = DepthwiseSeparableConv(96, 128, stride=2)                # 64x64
        self.enc_ds_conv3 = DepthwiseSeparableConv(128, 128, stride=1)               # 64x64
        self.enc_ds_conv4 = DepthwiseSeparableConv(128, 256, stride=2)               # 32x32
        self.enc_ds_conv5 = DepthwiseSeparableConv(256, 256, stride=1)               # 32x32
        self.enc_ds_conv6 = DepthwiseSeparableConv(256, 512, stride=2)               # 16x16
        
        # Улучшенные блоки внимания для skip-connections
        self.att_skip1 = QuantFriendlyAttention(96)
        self.att_skip2 = QuantFriendlyAttention(128)
        self.att_skip3 = QuantFriendlyAttention(256)
        
        # Улучшенные блоки обработки признаков на skip-connections
        self.edge_aware1 = EdgeAwareFeatureEnhancement(96)
        self.edge_aware2 = EdgeAwareFeatureEnhancement(128)
        self.edge_aware3 = EdgeAwareFeatureEnhancement(256)
        
        # Decoder с улучшенным слиянием признаков
        self.up1 = UpBlock(512, 256, scale_factor=2)                                # 16x16 -> 32x32
        self.fusion1 = FeatureFusion(256, 256, 256)
        
        self.up2 = UpBlock(256, 128, scale_factor=2)                                # 32x32 -> 64x64
        self.fusion2 = FeatureFusion(128, 128, 128)
        
        self.up3 = UpBlock(128, 96, scale_factor=2)                                 # 64x64 -> 128x128
        self.fusion3 = FeatureFusion(96, 96, 96)
        
        self.up4 = UpBlock(96, 48, scale_factor=2)                                  # 128x128 -> 256x256
        
        # Модуль улучшения деталей перед финальной сверткой
        self.detail_enhance = DetailEnhancementModule(48)
        
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
                # Ограничиваем начальные веса, чтобы избежать экстремальных значений
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
        enc1 = self.enc_conv1(x)          # 48 x 128 x 128
        skip1 = self.enc_ds_conv1(enc1)   # 96 x 128 x 128
        
        enc2 = self.enc_ds_conv2(skip1)   # 128 x 64 x 64
        skip2 = self.enc_ds_conv3(enc2)   # 128 x 64 x 64
        
        enc3 = self.enc_ds_conv4(skip2)   # 256 x 32 x 32
        skip3 = self.enc_ds_conv5(enc3)   # 256 x 32 x 32
        
        enc4 = self.enc_ds_conv6(skip3)   # 512 x 16 x 16
        
        # Улучшение признаков на skip-connections
        skip1 = self.edge_aware1(skip1)
        skip2 = self.edge_aware2(skip2)
        skip3 = self.edge_aware3(skip3)
        
        # Attention modules
        skip1_att = self.att_skip1(skip1)
        skip2_att = self.att_skip2(skip2)
        skip3_att = self.att_skip3(skip3)
        
        # Decoder с улучшенным слиянием признаков
        d1 = self.up1(enc4)                             # 256 x 32 x 32
        d1 = self.fusion1(d1, skip3_att)                # 256 x 32 x 32
        
        d2 = self.up2(d1)                               # 128 x 64 x 64
        d2 = self.fusion2(d2, skip2_att)                # 128 x 64 x 64
        
        d3 = self.up3(d2)                               # 96 x 128 x 128
        d3 = self.fusion3(d3, skip1_att)                # 96 x 128 x 128
        
        d4 = self.up4(d3)                               # 48 x 256 x 256
        
        # Улучшение деталей
        d4 = self.detail_enhance(d4)                    # 48 x 256 x 256
        
        # Финальные свертки
        out = self.final_conv1(d4)                      # 24 x 256 x 256
        depth = self.final_conv2(out)                   # 1 x 256 x 256
        
        # Используем sigmoid для плавной активации вместо hardtanh
        depth = torch.sigmoid(depth)
        
        return depth

if __name__ == "__main__":
    # Проверка модели
    model = ImprovedQuantizationFriendlyMobileDepth(input_size=(256, 256))
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Печать информации о модели
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Тест прямого прохода
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
