@echo off
echo ===================================
echo    MobileDepth ONNX Conversion
echo ===================================

REM Set paths
set MODEL_PATH=MidAir_optimized\checkpoints\best_model.pth
set OUTPUT_DIR=onnx_models

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Activate Python environment
call .\depth\Scripts\activate.bat

REM Check and install dependencies
echo Checking dependencies...
python -c "import onnx" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ONNX package not found. Installing required packages...
    pip install onnx onnxruntime protobuf
) else (
    echo ONNX package found.
)

REM Standard ONNX conversion
echo Converting model to ONNX format...
python convert_to_onnx.py --model_path %MODEL_PATH% --output_path %OUTPUT_DIR%\mobiledepth.onnx --input_size 384 --device cpu

REM Check if the first conversion was successful
if exist %OUTPUT_DIR%\mobiledepth.onnx (
    echo First conversion successful! Proceeding with optimized version...
    
    REM Optimized and quantized version for edge devices
    echo Creating optimized and quantized version for edge deployment...
    python convert_to_onnx.py --model_path %MODEL_PATH% --output_path %OUTPUT_DIR%\mobiledepth_optimized.onnx --input_size 384 --device cpu --optimize
    
    REM Create version with dynamic input dimensions
    echo Creating version with dynamic input dimensions...
    python convert_to_onnx.py --model_path %MODEL_PATH% --output_path %OUTPUT_DIR%\mobiledepth_dynamic.onnx --input_size 384 --device cpu --dynamic
    
    echo.
    echo Conversion completed! Models saved to: %OUTPUT_DIR%
    echo.
    echo Models ready for deployment on LicheePi4A with TH1520 NPU
) else (
    echo Error: Initial conversion failed. Please check the error messages above.
)

pause
