import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from bdepth_model import B_MobileDepth

def load_model(model_path, device='cpu'):
    """Load a trained B_MobileDepth model from a checkpoint"""
    model = B_MobileDepth(input_size=(320, 320))
    
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
    """Convert PyTorch model to ONNX format
    
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
        opset_version=11,
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
        
        # Create a random input tensor
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        output = session.run([output_name], {input_name: input_data})
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            print(f"Inference time: {elapsed:.2f} ms")
        
        # Print output shape
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

def optimize_onnx_for_inference(onnx_path, optimized_path=None):
    """Optimize ONNX model for inference
    
    Args:
        onnx_path: Path to the original ONNX model
        optimized_path: Path to save the optimized model (if None, will use the original path)
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        optimized_path = optimized_path or onnx_path
        
        print(f"Optimizing ONNX model for inference: {onnx_path}")
        
        # Load the model
        model = onnx.load(onnx_path)
        
        # Create optimization options
        opt_options = optimizer.OptimizationOptions()
        
        # Optimize the model
        opt_model = optimizer.optimize_model(
            onnx_path,
            'b_mobiledepth_opt',
            model_type='mobilenet',
            num_heads=0,
            hidden_size=0,
            optimization_options=opt_options
        )
        
        # Save the optimized model
        opt_model.save_model_to_file(optimized_path)
        
        print(f"Optimized ONNX model saved to: {optimized_path}")
        return optimized_path
    except ImportError:
        print("ONNX Runtime not found. Install with 'pip install onnxruntime-tools' to optimize the model.")
        return onnx_path
    except Exception as e:
        print(f"Error optimizing ONNX model: {e}")
        return onnx_path

def quantize_onnx_model(onnx_path, quantized_path=None):
    """Quantize ONNX model to INT8 for faster inference on edge devices
    
    Args:
        onnx_path: Path to the original ONNX model
        quantized_path: Path to save the quantized model (if None, will use {onnx_path_without_extension}_quantized.onnx)
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantized_path is None:
            base_path = os.path.splitext(onnx_path)[0]
            quantized_path = f"{base_path}_quantized.onnx"
        
        print(f"Quantizing ONNX model: {onnx_path}")
        
        # Quantize model to INT8
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8
        )
        
        print(f"Quantized ONNX model saved to: {quantized_path}")
        return quantized_path
    except ImportError:
        print("ONNX Runtime Quantization not found. Install with 'pip install onnxruntime-tools' to quantize the model.")
        return onnx_path
    except Exception as e:
        print(f"Error quantizing ONNX model: {e}")
        return onnx_path

def create_onnx_info_file(onnx_path, model_info, save_path=None):
    """Create a readme file with model information for deployment
    
    Args:
        onnx_path: Path to the ONNX model
        model_info: Dictionary with model information
        save_path: Path to save the info file (if None, will use the same directory as the ONNX model)
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(onnx_path), "model_info.txt")
    
    # Try to get model size
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024) if os.path.exists(onnx_path) else "Unknown"
    
    # Get model details if possible
    input_shape = "Unknown"
    output_shape = "Unknown"
    try:
        import onnx
        model = onnx.load(onnx_path)
        if model.graph.input and model.graph.input[0].type.tensor_type.shape.dim:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            input_shape = "x".join([str(dim.dim_value if dim.dim_value else "?") for dim in dims])
        
        if model.graph.output and model.graph.output[0].type.tensor_type.shape.dim:
            dims = model.graph.output[0].type.tensor_type.shape.dim
            output_shape = "x".join([str(dim.dim_value if dim.dim_value else "?") for dim in dims])
    except:
        pass
    
    with open(save_path, "w") as f:
        f.write("# B_MobileDepth ONNX Model Information\n\n")
        f.write(f"Date: {model_info.get('date', 'Unknown')}\n")
        f.write(f"Original PyTorch model: {model_info.get('pytorch_model', 'Unknown')}\n")
        f.write(f"ONNX model: {os.path.basename(onnx_path)}\n")
        f.write(f"Model size: {model_size_mb:.2f} MB\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Output shape: {output_shape}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"Validation loss: {model_info.get('val_loss', 'Unknown')}\n")
        f.write(f"AbsRel: {model_info.get('abs_rel', 'Unknown')}\n")
        f.write(f"RMSE: {model_info.get('rmse', 'Unknown')}\n\n")
        
        f.write("## Usage Instructions for LicheePi4A TH1520 NPU\n\n")
        f.write("1. Deploy this ONNX model to your LicheePi4A device\n")
        f.write("2. Use ONNX Runtime or TH1520 NPU SDK to run inference\n")
        f.write("3. Input should be a normalized RGB image with values in range [0,1]\n")
        f.write("4. Output is a depth map with values in range [0,1]\n\n")
        
        f.write("## Pre-processing\n\n")
        f.write("```python\n")
        f.write("from PIL import Image\n")
        f.write("import numpy as np\n\n")
        f.write("# Load and preprocess image\n")
        f.write(f"img = Image.open('image.jpg').convert('RGB').resize(({model_info.get('image_size', 320)}, {model_info.get('image_size', 320)}))\n")
        f.write("img_np = np.array(img).astype(np.float32) / 255.0\n")
        f.write("# Normalize with ImageNet mean and std\n")
        f.write("mean = np.array([0.485, 0.456, 0.406])\n")
        f.write("std = np.array([0.229, 0.224, 0.225])\n")
        f.write("img_np = (img_np - mean) / std\n")
        f.write("# Transpose from HWC to CHW format\n")
        f.write("img_np = img_np.transpose(2, 0, 1)\n")
        f.write("# Add batch dimension\n")
        f.write("img_np = np.expand_dims(img_np, axis=0)\n")
        f.write("```\n\n")
        
        f.write("## Post-processing\n\n")
        f.write("```python\n")
        f.write("# Process output depth map\n")
        f.write("depth_map = output[0][0]  # Remove batch and channel dimensions\n")
        f.write("# Normalize to 0-255 for visualization\n")
        f.write("depth_min = depth_map.min()\n")
        f.write("depth_max = depth_map.max()\n")
        f.write("depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)\n")
        f.write("depth_viz = (depth_norm * 255).astype(np.uint8)\n")
        f.write("# Apply colormap for better visualization\n")
        f.write("import cv2\n")
        f.write("depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_PLASMA)\n")
        f.write("```\n")
    
    print(f"Model information saved to: {save_path}")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Convert B_MobileDepth PyTorch model to ONNX format")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained PyTorch model checkpoint")
    parser.add_argument("--output_path", type=str, default="onnx_models/bmobiledepth.onnx",
                        help="Path to save the ONNX model")
    
    # Conversion parameters
    parser.add_argument("--input_size", type=int, default=320,
                        help="Input image size (assumed square)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic input size")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for conversion (cpu or cuda)")
    
    # Optimization options
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize ONNX model for inference")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize ONNX model to INT8")
    parser.add_argument("--test_inference", action="store_true",
                        help="Test inference with the exported ONNX model")
    
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
        print("Or run the provided install_dependencies.bat script")
        return
    
    # Set device
    device = args.device if args.device in ["cpu", "cuda"] and (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(args.model_path, device=device)
        
        # Extract model information
        import datetime
        model_info = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "pytorch_model": args.model_path,
            "image_size": args.input_size
        }
        
        # Get validation metrics from checkpoint
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'val_loss' in checkpoint:
                model_info['val_loss'] = checkpoint['val_loss']
            if 'abs_rel' in checkpoint:
                model_info['abs_rel'] = checkpoint['abs_rel']
            if 'rmse' in checkpoint:
                model_info['rmse'] = checkpoint['rmse']
        except:
            print("Could not extract metrics from checkpoint")
        
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
        
        # Optimize if requested
        if args.optimize:
            try:
                onnx_path = optimize_onnx_for_inference(onnx_path)
            except Exception as e:
                print(f"Warning: Optimization failed: {e}")
                print("Continuing with the non-optimized model.")
        
        # Quantize if requested
        if args.quantize:
            try:
                base_path = os.path.splitext(onnx_path)[0]
                quantized_path = f"{base_path}_quantized.onnx"
                quantize_onnx_model(onnx_path, quantized_path)
            except Exception as e:
                print(f"Warning: Quantization failed: {e}")
                print("Continuing with the non-quantized model.")
        
        # Test inference if requested
        if args.test_inference:
            try:
                test_onnx_inference(onnx_path, input_shape=input_shape)
            except Exception as e:
                print(f"Warning: Inference testing failed: {e}")
        
        # Create model info file
        create_onnx_info_file(onnx_path, model_info)
        
        print("\nConversion completed successfully!")
        print(f"ONNX model saved to: {os.path.abspath(onnx_path)}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
