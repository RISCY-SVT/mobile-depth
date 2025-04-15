import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from depth_model import ImprovedQuantizationFriendlyMobileDepth

def load_model(model_path, input_size=(256, 256), device='cpu'):
    """Load a trained ImprovedQuantizationFriendlyMobileDepth model from a checkpoint"""
    model = ImprovedQuantizationFriendlyMobileDepth(input_size=input_size)
    
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check what's in the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_loss' in checkpoint:
                print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
            if 'abs_rel' in checkpoint:
                print(f"Checkpoint AbsRel: {checkpoint['abs_rel']:.4f}")
            if 'rmse' in checkpoint:
                print(f"Checkpoint RMSE: {checkpoint['rmse']:.4f}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    return model

def convert_to_onnx(model, output_path, input_shape=(1, 3, 256, 256), device='cpu', dynamic=False):
    """Convert PyTorch model to ONNX format optimized for TH1520 NPU
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        device: Device to run the conversion on
        dynamic: Whether to use dynamic input size
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define the input tensor
    dummy_input = torch.randn(input_shape, device=device)
    
    # Normalize input to simulate real inference conditions
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    dummy_input = (dummy_input - mean) / std
    
    # Get model input and output names
    input_names = ["input"]
    output_names = ["output"]
    
    # Define dynamic axes if requested
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    
    # Export the model to ONNX format
    print(f"Converting model to ONNX format with input shape: {input_shape}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Используем opset 11 для лучшей совместимости с TH1520
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=False,
        verbose=False
    )
    
    print(f"ONNX model saved to: {output_path}")
    print("Verifying ONNX model...")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed!")
        
        # Print model metadata
        metadata = {}
        for meta in onnx_model.metadata_props:
            metadata[meta.key] = meta.value
        
        print(f"Model IR version: {onnx_model.ir_version}")
        print(f"Opset version: {onnx_model.opset_import[0].version}")
        print(f"Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        if metadata:
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    except ImportError:
        print("ONNX package not found. Install with 'pip install onnx' to verify the model.")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
    
    return output_path

def test_onnx_inference(onnx_path, input_shape=(1, 3, 256, 256)):
    """Test inference with the exported ONNX model
    
    Args:
        onnx_path: Path to the ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    try:
        import onnxruntime as ort
        
        print(f"Testing ONNX inference with input shape: {input_shape}")
        
        # Create an ONNX Runtime session
        print("Available providers:", ort.get_available_providers())
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Print input and output details
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Model Input: {input_name}, shape: {session.get_inputs()[0].shape}")
        print(f"Model Output: {output_name}, shape: {session.get_outputs()[0].shape}")
        
        # Create a normalized input tensor (simulating real input)
        input_data = np.random.rand(*input_shape).astype(np.float32)
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        input_data = (input_data - mean) / std
        
        # Run inference
        output = session.run([output_name], {input_name: input_data})
        
        # Print output shape and statistics
        output_data = output[0]
        print(f"Output shape: {output_data.shape}")
        print(f"Output min: {output_data.min()}, max: {output_data.max()}, mean: {output_data.mean()}")
        
        return True
    except ImportError:
        print("ONNX Runtime not found. Install with 'pip install onnxruntime' to test inference.")
        return False
    except Exception as e:
        print(f"Error testing ONNX inference: {e}")
        return False

def create_hhb_conversion_script(onnx_path, output_dir):
    """Create a script to convert ONNX model to TH1520 NPU binary using HHB
    
    Args:
        onnx_path: Path to the ONNX model
        output_dir: Directory to save the script
    """
    script_path = os.path.join(output_dir, "convert_onnx_to_hhb.sh")
    
    script_content = f'''#!/bin/bash
# Скрипт для конвертации ONNX модели в формат для TH1520 NPU с использованием HHB

# Выходной каталог
OUTPUT_DIR="./npu_model"
mkdir -p $OUTPUT_DIR

# Путь к ONNX модели
ONNX_MODEL="{os.path.basename(onnx_path)}"

# Проверка наличия изображений для калибровки
if [ ! -d "./calibration_images" ] || [ $(ls -1 ./calibration_images | wc -l) -lt 10 ]; then
    echo "Нужно минимум 10 изображений для хорошей калибровки."
    mkdir -p ./calibration_images
    # Если нет изображений, скопируйте сюда не менее 10 изображений для калибровки
fi

# Запуск HHB для конвертации
hhb -v -v -v -v -D \\
    --board th1520 \\
    --link-lib shl_th1520 \\
    --model-format onnx \\
    --quantization-scheme "int8_sym" \\
    --quantization-loss-threshold 0.999 \\
    --model-file "$ONNX_MODEL" \\
    --data-scale-div 255 \\
    --input-name "input" \\
    --output-name "output" \\
    --input-shape "1 3 256 256" \\
    --calibrate-dataset "./calibration_images" \\
    --output "$OUTPUT_DIR" \\
    2>&1 | tee hhb_conversion.log

echo "Конвертация завершена. Результаты в $OUTPUT_DIR"
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"HHB conversion script created: {script_path}")
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Convert MobileDepth PyTorch model to ONNX format for TH1520 NPU")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained PyTorch model checkpoint")
    parser.add_argument("--output_path", type=str, default="onnx_models/mobiledepth.onnx",
                        help="Path to save the ONNX model")
    
    # Conversion parameters
    parser.add_argument("--input_size", type=int, default=256,
                        help="Input image size (assumed square)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic input size")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for conversion (cpu or cuda)")
    
    # Optimization options
    parser.add_argument("--test_inference", action="store_true",
                        help="Test inference with the exported ONNX model")
    parser.add_argument("--create_hhb_script", action="store_true", default=True,
                        help="Create script for HHB conversion")
    
    args = parser.parse_args()
    
    # Check for required dependencies
    missing_deps = []
    try:
        import onnx
    except ImportError:
        missing_deps.append("onnx")
    
    try:
        import onnxruntime
    except ImportError:
        missing_deps.append("onnxruntime")
    
    if missing_deps:
        print(f"Error: Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install them with: pip install " + " ".join(missing_deps))
        return
    
    # Set device
    device = args.device if args.device in ["cpu", "cuda"] and (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load model
        input_size = (args.input_size, args.input_size)
        model = load_model(args.model_path, input_size=input_size, device=device)
        
        # Ensure model is in eval mode
        model.eval()
        
        # Extract model information
        import datetime
        model_info = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "pytorch_model": args.model_path,
            "input_size": input_size
        }
        
        # Convert to ONNX
        input_shape = (args.batch_size, 3, args.input_size, args.input_size)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        onnx_path = convert_to_onnx(
            model,
            args.output_path,
            input_shape=input_shape,
            device=device,
            dynamic=args.dynamic
        )
        
        # Test inference if requested
        if args.test_inference:
            try:
                test_onnx_inference(onnx_path, input_shape=input_shape)
            except Exception as e:
                print(f"Warning: Inference testing failed: {e}")
        
        # Create HHB conversion script if requested
        if args.create_hhb_script:
            try:
                output_dir = os.path.dirname(os.path.abspath(args.output_path))
                create_hhb_conversion_script(onnx_path, output_dir)
            except Exception as e:
                print(f"Warning: Failed to create HHB script: {e}")
        
        print("\nConversion completed successfully!")
        print(f"ONNX model saved to: {os.path.abspath(onnx_path)}")
        print("\nNext steps for deployment to LicheePi4A TH1520 NPU:")
        print("1. Transfer the ONNX model to your LicheePi4A device")
        print("2. Create calibration images for quantization")
        print("3. Run the generated HHB conversion script")
        print("4. Use the resulting model with the TH1520 NPU SDK")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
