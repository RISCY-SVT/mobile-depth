#!/usr/bin/env bash

python3 convert_to_onnx.py \
  --model_path MidAir_enhanced/checkpoints/best_model.pth \
  --output_path onnx_models/mobiledepth.onnx \
  --input_size 256 \
  --device cpu \
  --test_inference \
  --create_hhb_script
