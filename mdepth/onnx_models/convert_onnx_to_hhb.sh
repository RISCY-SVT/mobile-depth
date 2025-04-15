#!/usr/bin/env bash
# Скрипт для конвертации ONNX модели в формат для TH1520 NPU с использованием HHB

# Выходной каталог
OUTPUT_DIR="./npu_model"
mkdir -p $OUTPUT_DIR

# Путь к ONNX модели
ONNX_MODEL="mobiledepth.onnx"

# Проверка наличия изображений для калибровки
if [[ ! -d ../calibration_images ]] || [[ $(ls -1 ../calibration_images | wc -l) -lt 10 ]]; then
    echo "###-WARNING((line $LINENO): Нужно минимум 10 изображений для хорошей калибровки."
    mkdir -p ../calibration_images
    # Если нет изображений, скопируйте сюда не менее 10 изображений для калибровки
fi

# Запуск HHB для конвертации
hhb -v -v -v -v -D \
    --board th1520 \
    --link-lib shl_th1520 \
    --model-format onnx \
    --quantization-scheme "int8_sym" \
    --quantization-loss-threshold 0.999 \
    --model-file "$ONNX_MODEL" \
    --data-scale-div 255 \
    --input-name "input" \
    --output-name "output" \
    --input-shape "1 3 256 256" \
    --calibrate-dataset ../calibration_images \
    --output "$OUTPUT_DIR" \
    2>&1 | tee onnx_to_hhb_$(date +%Y-%m-%d_%H-%M-%S).log

echo "Конвертация завершена. Результаты в $OUTPUT_DIR"

