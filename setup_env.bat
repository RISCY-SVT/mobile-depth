@echo off
REM Создаём виртуальное окружение
python -m venv bdepth

REM Активируем окружение (важно использовать "call", чтобы скрипт не завершался после активации)
call .\bdepth\Scripts\activate.bat

REM Обновляем pip
python -m pip install --upgrade pip

REM Устанавливаем PyTorch с поддержкой CUDA и другие пакеты по необходимости
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib pillow tqdm psutil onnxruntime onnx opencv_python pandas
pip cache purge
