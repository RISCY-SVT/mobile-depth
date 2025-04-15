import torch
import torch.nn as nn
import torch.nn.functional as F

# Базовый сверточный блок с BatchNorm и активацией (ReLU6 по умолчанию)
class ConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU6):
        super(ConvBNReLU6, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
            activation(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# Блок depthwise-separable свертки
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation=nn.ReLU6):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5),
            activation(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
            activation(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Пространственное внимание (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

# Улучшенный блок апсемплинга с остаточным соединением
class ImprovedUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, activation=nn.ReLU6):
        super(ImprovedUpBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = ConvBNReLU6(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation)
        self.conv2 = ConvBNReLU6(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=activation)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        shortcut = self.shortcut(x_up)
        out = self.conv1(x_up)
        out = self.conv2(out)
        return out + shortcut

# Основная сеть MobileDepth с encoder-decoder, вниманием, мульти-масштабными выходами и улучшенной инициализацией
class MobileDepth(nn.Module):
    def __init__(self, num_classes=1, activation=nn.ReLU6):
        super(MobileDepth, self).__init__()
        # Encoder (MobileNet-v1)
        self.initial_conv = ConvBNReLU6(3, 32, kernel_size=3, stride=2, padding=1, activation=activation)  # [B,32,320,320]
        self.dsconv1 = DepthwiseSeparableConv(32, 64, stride=1, activation=activation)                     # [B,64,320,320]
        self.dsconv2 = DepthwiseSeparableConv(64, 128, stride=2, activation=activation)                     # [B,128,160,160]
        self.dsconv3 = DepthwiseSeparableConv(128, 128, stride=1, activation=activation)                    # [B,128,160,160]
        self.dsconv4 = DepthwiseSeparableConv(128, 256, stride=2, activation=activation)                    # [B,256,80,80]
        self.dsconv5 = DepthwiseSeparableConv(256, 256, stride=1, activation=activation)                    # [B,256,80,80]
        self.dsconv6 = DepthwiseSeparableConv(256, 512, stride=2, activation=activation)                    # [B,512,40,40]
        
        # Skip connections:
        # skip0: output initial_conv -> [B,32,320,320]
        # skip1: output dsconv1 -> [B,64,320,320]
        # skip2: output dsconv3 -> [B,128,160,160]
        # skip3: output dsconv5 -> [B,256,80,80]
        
        # Attention modules for skip connections
        self.att_skip1 = SpatialAttention(64)
        self.att_skip2 = SpatialAttention(128)
        self.att_skip3 = SpatialAttention(256)
        
        # Decoder with improved up blocks
        self.up1 = ImprovedUpBlock(512, 256, scale_factor=2, activation=activation)  # d1: [B,256,80,80]
        self.up2 = ImprovedUpBlock(512, 128, scale_factor=2, activation=activation)  # input: cat(d1, att(skip3)) -> [B,256+256=512,80,80] -> [B,128,160,160]
        self.up3 = ImprovedUpBlock(256, 64, scale_factor=2, activation=activation)   # input: cat(d2, att(skip2)) -> [B,128+128=256,160,160] -> [B,64,320,320]
        self.up4 = ImprovedUpBlock(128, 32, scale_factor=2, activation=activation)   # input: cat(d3, att(skip1)) -> [B,64+64=128,320,320] -> [B,32,640,640]
        
        # Final conv for full-resolution output
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        # Дополнительные свёрточные слои для мульти-масштабного выхода
        self.final_conv_d3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.final_conv_d2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        skip0 = self.initial_conv(x)       # [B,32,320,320]
        skip1 = self.dsconv1(skip0)        # [B,64,320,320]
        x1 = self.dsconv2(skip1)           # [B,128,160,160]
        skip2 = self.dsconv3(x1)           # [B,128,160,160]
        x2 = self.dsconv4(skip2)           # [B,256,80,80]
        skip3 = self.dsconv5(x2)           # [B,256,80,80]
        x3 = self.dsconv6(skip3)           # [B,512,40,40]
        
        # Decoder с вниманием
        d1 = self.up1(x3)                # [B,256,80,80]
        d1 = torch.cat([d1, self.att_skip3(skip3)], dim=1)  # [B,256+256=512,80,80]
        d2 = self.up2(d1)                # [B,128,160,160]
        d2 = torch.cat([d2, self.att_skip2(skip2)], dim=1)  # [B,128+128=256,160,160]
        d3 = self.up3(d2)                # [B,64,320,320]
        d3 = torch.cat([d3, self.att_skip1(skip1)], dim=1)  # [B,64+64=128,320,320]
        d4 = self.up4(d3)                # [B,32,640,640]
        
        # Мульти-масштабные выходы
        out1 = self.final_conv(d4)        # [B,1,640,640]
        out2 = F.interpolate(self.final_conv_d3(d3), scale_factor=2, mode='bilinear', align_corners=True)  # [B,1,640,640]
        out3 = F.interpolate(self.final_conv_d2(d2), scale_factor=4, mode='bilinear', align_corners=True)  # [B,1,640,640]
        
        # Применяем сигмоид для нормализации выходов (если target в [0,1])
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        
        if self.training:
            return [out1, out2, out3]
        else:
            return out1

if __name__ == "__main__":
    model = MobileDepth(num_classes=1)
    # Тестируем с входным изображением 1024x1024 (можно адаптировать под нужное разрешение)
    x = torch.randn(1, 3, 1024, 1024)
    outputs = model(x)
    if isinstance(outputs, list):
        print("Output shapes:", [out.shape for out in outputs])
    else:
        print("Output shape:", outputs.shape)
