#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования модели MobileDepth в формате ONNX
"""

import argparse
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Test ONNX model for depth estimation")
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output visualization')
    parser.add_argument('--input_size', type=int, default=256, help='Input size for the model')
    parser.add_argument('--color_map', type=str, default='plasma', 
                        choices=['plasma', 'magma', 'inferno', 'viridis', 'jet'],
                        help='Colormap for depth visualization')
    parser.add_argument('--verbose', action='store_true', help='Print verbose information')
    return parser.parse_args()


def load_onnx_model(model_path):
    """Загрузка ONNX модели"""
    print(f"Loading ONNX model from {model_path}")
    # Создаем сессию ONNX Runtime
    try:
        session = ort.InferenceSession(model_path)
        print("Model loaded successfully!")
        
        # Вывод информации о входах и выходах модели
        print("\nModel inputs:")
        for i, input_tensor in enumerate(session.get_inputs()):
            print(f"  {i}: name={input_tensor.name}, shape={input_tensor.shape}, type={input_tensor.type}")
        
        print("\nModel outputs:")
        for i, output_tensor in enumerate(session.get_outputs()):
            print(f"  {i}: name={output_tensor.name}, shape={output_tensor.shape}, type={output_tensor.type}")
            
        return session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None


def preprocess_image(image_path, input_size):
    """Предварительная обработка изображения"""
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size
    
    # Изменение размера изображения
    img = img.resize((input_size, input_size), Image.BILINEAR)
    
    # Преобразование в numpy и нормализация
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Нормализация ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_np = (img_np - mean) / std
    
    # Преобразование в NCHW формат
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np, (orig_width, orig_height)


def run_inference(session, input_data):
    """Запуск инференса модели"""
    print("Running inference...")
    start_time = time.time()
    
    # Получаем имя входного тензора
    input_name = session.get_inputs()[0].name
    
    # Запускаем инференс
    try:
        outputs = session.run(None, {input_name: input_data})
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f} seconds")
        
        return outputs, inference_time
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None


def visualize_depth(rgb_image, depth_map, output_path, original_size=None, colormap='plasma'):
    """Визуализация карты глубины"""
    print(f"Visualizing depth map and saving to {output_path}")
    
    # Преобразование из NCHW в HWC
    rgb = rgb_image[0].transpose(1, 2, 0)
    
    # Если глубина имеет 4D формат [1, 1, H, W], преобразуем в 2D [H, W]
    if len(depth_map.shape) == 4:
        depth = depth_map[0, 0]
    elif len(depth_map.shape) == 3:
        depth = depth_map[0]
    else:
        depth = depth_map
    
    # Изменение размера до оригинального, если предоставлен
    if original_size:
        rgb_resized = cv2.resize(rgb, original_size, interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, original_size, interpolation=cv2.INTER_LINEAR)
    else:
        rgb_resized = rgb
        depth_resized = depth
    
    # Нормализация глубины для визуализации
    if depth_resized.min() != depth_resized.max():
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())
    else:
        depth_normalized = depth_resized
    
    # Создание визуализации
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Исходное RGB изображение
    axs[0, 0].imshow(rgb_resized)
    axs[0, 0].set_title('RGB Input')
    axs[0, 0].axis('off')
    
    # Карта глубины с цветовой картой
    depth_colored = axs[0, 1].imshow(depth_normalized, cmap=colormap)
    axs[0, 1].set_title(f'Depth Map (min={depth_normalized.min():.2f}, max={depth_normalized.max():.2f})')
    axs[0, 1].axis('off')
    plt.colorbar(depth_colored, ax=axs[0, 1])
    
    # Гистограмма глубины
    axs[1, 0].hist(depth_normalized.flatten(), bins=100)
    axs[1, 0].set_title('Depth Histogram')
    axs[1, 0].set_xlabel('Normalized Depth')
    axs[1, 0].set_ylabel('Pixel Count')
    
    # 3D визуализация (псевдо-3D с помощью заполнения цветов)
    axs[1, 1].imshow(depth_normalized, cmap=colormap)
    axs[1, 1].set_title('Pseudo-3D Visualization')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Дополнительно сохраняем саму карту глубины в формате PNG
    depth_for_save = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_for_save, cv2.COLORMAP_INFERNO)
    cv2.imwrite(output_path.replace('.png', '_depth.png'), depth_colored)
    print(f"Depth map saved to {output_path.replace('.png', '_depth.png')}")


def main():
    args = parse_args()
    
    # Загрузка модели
    model = load_onnx_model(args.model)
    if model is None:
        return
    
    # Предобработка изображения
    input_data, original_size = preprocess_image(args.image, args.input_size)
    if input_data is None:
        return
    
    # Инференс
    outputs, inference_time = run_inference(model, input_data)
    if outputs is None:
        return
    
    # Выводим информацию о результатах
    for i, output in enumerate(outputs):
        if args.verbose:
            print(f"Output {i}:")
            print(f"  Shape: {output.shape}")
            print(f"  Range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  Mean: {output.mean():.6f}")
            print(f"  Standard deviation: {output.std():.6f}")
    
    # Визуализация результата (используем первый выход)
    depth_map = outputs[0]
    visualize_depth(input_data, depth_map, args.output, original_size, args.color_map)


if __name__ == "__main__":
    main()
