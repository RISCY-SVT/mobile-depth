#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converter for your depth model to ONNX format optimized for LicheePi4A.
"""

import torch
import numpy as np
import torch.nn.functional as F
import os
from depth_model import MobileDepth 

def run(model_path, input_size=256):
    print(f"Initializing model from: {model_path}")
    
    # Загрузка вашей модели
    model = MobileDepth()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"Starting processing with input size {input_size}x{input_size}")
    
    # Определение входного тензора (3 x input_size x input_size)
    dummy_input = np.zeros((3, input_size, input_size), np.float32)
    sample = torch.from_numpy(dummy_input).unsqueeze(0)
    
    # (Опционально) Тестовый проход для проверки выходных данных
    with torch.no_grad():
        prediction = model(sample)
        # Если ваша модель не делает интерполяцию до входного размера, добавьте:
        if prediction.dim() == 3:  # Если выход имеет форму [B, H, W]
            prediction = prediction.unsqueeze(1)  # Добавить канал [B, 1, H, W]
        
        prediction = F.interpolate(
            prediction,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
        )
        
        if prediction.dim() == 4 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)  # Удалить канал для глубины [B, H, W]
        
        prediction = torch.clamp(prediction, min=0)  # non_negative
        result = prediction.squeeze().cpu().numpy()
        print(f"Test output shape: {result.shape}, min: {result.min()}, max: {result.max()}")
    
    # Создание имени ONNX файла на основе имени файла модели
    onnx_filename = os.path.basename(model_path).rsplit('.', 1)[0] + f'_{input_size}.onnx'
    
    # Экспорт модели в ONNX
    torch.onnx.export(
        model,
        sample,
        onnx_filename,
        opset_version=11,  # Используем более консервативную версию
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Только батч динамический
            "output": {0: "batch_size"}  # Только батч динамический
        }
    )
    
    print("Export finished. Saved as:", onnx_filename)

if __name__ == "__main__":
    # Укажите путь к вашей модели
    MODEL_PATH = "MidAir_optimized/checkpoints/best_model.pth"
    run(MODEL_PATH, input_size=256)  # Используем размер 256x256
