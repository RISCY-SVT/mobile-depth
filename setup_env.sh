#!/bin/bash
# Создание виртуального окружения
python3 -m venv bdepth

# Активация виртуального окружения
source bdepth/bin/activate

# Обновление pip
python -m pip install --upgrade pip

# Установка PyTorch с поддержкой CUDA и других пакетов
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib pillow tqdm psutil onnxruntime onnx opencv_python pandas
pip cache purge
