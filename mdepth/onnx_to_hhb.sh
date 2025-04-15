#!/usr/bin/env bash
set -eE
set -o pipefail

# Настройка переменных
MODEL_NAME="onnx_models/mobiledepth"
ONNX_MODEL="${MODEL_NAME}.onnx"
MODEL_INPUT="input"
MODEL_OUTPUT="output"
MODEL_INPUT_SHAPE="1 3 256 256"
OUTPUT_DIR="./npu_model"
CALIBRATION_DIR="./calibration_images"

# Создание директорий, если они не существуют
mkdir -p $OUTPUT_DIR
mkdir -p $CALIBRATION_DIR

# Убедитесь, что у вас есть минимум 100 изображений для калибровки
# Если нет, скопируйте пример изображения несколько раз
if [ ! -d "$CALIBRATION_DIR" ] || [ $(ls -1 $CALIBRATION_DIR | wc -l) -lt 10 ]; then
    echo "Необходимо минимум 10 изображений для хорошей калибровки."
    echo "Копируем образец 000021.JPEG для создания набора калибровки..."
    for i in {1..10}; do
        cp 000021.JPEG $CALIBRATION_DIR/calibration_$i.JPEG
    done
fi

# Выполнение конвертации с оптимальными настройками
            # --quantization-scheme {int4_asym_w_sym,
            #                        uint8_asym,
            #                        int8_asym,
            #                        int8_sym,
            #                        int8_original,
            #                        int8_asym_w_sym,
            #                        int16_sym,
            #                        float16,
            #                        float16_w_int8,
            #                        float32,
            #                        unset}
            #             Scheme of quantization. default is unset.
# hhb -v -v -v -v -D \
#     --board th1520 \
#     --link-lib shl_th1520 \
#     --model-format onnx \
#     --quantization-scheme "int8_sym" \
#     --quantization-loss-threshold 0.999 \
#     --model-file "$ONNX_MODEL" \
#     --data-scale-div 255 \
#     --input-name "$MODEL_INPUT" \
#     --output-name "$MODEL_OUTPUT" \
#     --input-shape "$MODEL_INPUT_SHAPE" \
#     --calibrate-dataset "$CALIBRATION_DIR" \
#     --output "$OUTPUT_DIR" \
#     2>&1 | tee hhb_conversion.log

# hhb -v -v -v -v -D \
#     --board th1520 \
#     --link-lib shl_th1520 \
#     --model-format onnx \
#     --quantization-scheme "int16_sym" \    # Переход на 16-бит для большей точности
#     --quantization-loss-threshold 0.995 \  # Более строгий порог потери точности
#     --auto-hybrid-quantization \           # Автоматическая смешанная точность
#     --quantization-loss-algorithm "mse" \  # MSE для оценки потери точности
#     --model-file "$ONNX_MODEL" \
#     --data-scale-div 255 \
#     --input-name "$MODEL_INPUT" \
#     --output-name "$MODEL_OUTPUT" \
#     --input-shape "$MODEL_INPUT_SHAPE" \
#     --calibrate-dataset "$CALIBRATION_DIR" \
#     --output "$OUTPUT_DIR"


set -x
hhb -v -v -v -v -D \
    --board th1520 \
    --link-lib shl_th1520 \
    --model-format onnx \
    --quantization-scheme "float16" \
    --model-file "$ONNX_MODEL" \
    --data-scale-div 255 \
    --input-name "$MODEL_INPUT" \
    --output-name "$MODEL_OUTPUT" \
    --input-shape "$MODEL_INPUT_SHAPE" \
    --calibrate-dataset "$CALIBRATION_DIR" \
    --output "$OUTPUT_DIR"


# Проверка успешного завершения
if [ $? -eq 0 ]; then
    echo "Конвертация успешно завершена! Модель сохранена в $OUTPUT_DIR"
    echo "Созданные файлы:"
    ls -la $OUTPUT_DIR
else
    echo "Ошибка при конвертации. Проверьте лог hhb_conversion.log"
    exit 1
fi
chown svt:svt * -R

exit 0

###################################################################################################################################
# Removed:
    --fuse-conv-relu \
    --fuse-zp2bias \
    --quantization-loss-algorithm "mse" \
    --low-bound-scale 1.0 \
    --target-layout "NCHW" \
    --output-layout "NCHW" \
    --memory-type 2 \
    --model-priority 2 \
    --model-save "save_and_run" \


# From Midas/HHB
hhb -v -v -v -v -D \
--board th1520 \
--link-lib shl_th1520 \
--model-format onnx \
--model-file midas_v21_small_256.onnx \
--data-scale-div 255 \
--input-name "input" \
--output-name "output" \
--input-shape "1 3 256 256" \
--quantization-scheme "int8_asym" \
--calibrate-dataset kite.jpg \
2>&1 | tee hhb.log







#======================================================================================================================================================
usage: HHB [-E | -Q | -C | -D | -S | --simulate] [-v] [-h] [--version] [--no-quantize] [-f MODEL_FILE [MODEL_FILE ...]] [--save-temps] [--generate-dataset] [-lqm {q8_0,q4_0,q4_1,q4_k,q2_k,nf4_0,q8_0_fp32,q8_0_fp16,q4_0_fp32,q4_0_fp16}] [-lqc LLM_QUANT_CONFIG] [-lqr LLM_QUANT_RECIPE] [-in INPUT_NAME]
           [-is INPUT_SHAPE] [-on OUTPUT_NAME] [--model-format {keras,onnx,pb,tflite,pytorch,caffe}] [--reorder-pixel-format] [--board {th1520,e907,c906,c908,c920,c920v2,rvm,x86_ref,c907,c907rv32,unset}] [--hybrid-computing] [--link-lib {shl_pnna,shl_th1520,shl_c906,shl_c908,shl_c920,shl_rvm,unset}]
           [-cd CALIBRATE_DATASET] [-qs {int4_asym_w_sym,uint8_asym,int8_asym,int8_sym,int8_original,int8_asym_w_sym,int16_sym,float16,float16_w_int8,float32,unset}] [--auto-hybrid-quantization] [--quantization-loss-algorithm {cos_similarity,mse,kl_divergence,cross_entropy,gini}]
           [--quantization-loss-threshold QUANTIZATION_LOSS_THRESHOLD] [--dump-quantization-loss] [--hybrid-quantization-scheme {int4_asym_w_sym,uint8_asym,int8_asym,int8_sym,int8_asym_w_sym,int16_sym,float16,float32,unset}] [--hybrid-layer-name HYBRID_LAYER_NAME [HYBRID_LAYER_NAME ...]]
           [--low-bound-scale LOW_BOUND_SCALE] [--high-bound-scale HIGH_BOUND_SCALE] [--fuse-conv-relu] [--fuse-sigmoid-mul] [--fuse-reshape-dense] [--broadcast-quantization] [--fuse-clip] [--fuse-zp2bias] [--target-layout {NCHW,NHWC}] [--output-layout {NCHW,NHWC}] [--quantization-tool {default,ppq}] [--lsq]
           [--lsq-lr LSQ_LR] [--lsq-steps LSQ_STEPS] [--quant-device {cpu,cuda}] [--cali-batch CALI_BATCH] [--align-elementwise {None,Align to Large}] [--matrix-extension-mlen MATRIX_EXTENSION_MLEN] [-s DATA_SCALE | -sv DATA_SCALE_DIV] [-m DATA_MEAN] [-r DATA_RESIZE] [--pixel-format {RGB,BGR}]
           [--add-preprocess-node] [--config-file CONFIG_FILE] [--generate-config] [-o OUTPUT] [--trace {relay,qnn,csinn,csinn_acc} [{relay,qnn,csinn,csinn_acc} ...]] [-sd SIMULATE_DATA] [--postprocess {top5,save,save_and_top5}] [--show-session-run-time] [--model-save {run_only,save_only,save_and_run}]
           [--model-priority MODEL_PRIORITY] [--without-preprocess] [--input-memory-type {0,1,2} [{0,1,2} ...]] [--output-memory-type {0,1,2} [{0,1,2} ...]] [--memory-type {0,1,2}] [--ahead-of-time {intrinsic,unset}] [--dynamic-shape] [--device-thread DEVICE_THREAD]
           {'simulate', 'import', 'codegen', 'profiler', 'quantize'} ...

HHB command line tools

optional arguments:
  -E                    Convert model into relay ir.
  -Q                    Quantize the relay ir.
  -C                    codegen the model.
  -D                    deploy on platform.
  -S                    run elf to simulate.
  --simulate            Simulate model on x86 device.
  -v, --verbose         Increase verbosity
  -h, --help            Show this help information
  --version             Print the version and exit
  --no-quantize         If set, don't quantize the model.
  -f MODEL_FILE [MODEL_FILE ...], --model-file MODEL_FILE [MODEL_FILE ...]
                        Path to the input model file, can pass multi files
  --save-temps          Save temp files.
  --generate-dataset    Generate dataset according to provided preprocess parameters.
  -lqm {q8_0,q4_0,q4_1,q4_k,q2_k,nf4_0,q8_0_fp32,q8_0_fp16,q4_0_fp32,q4_0_fp16}, --llm-quant-mode {q8_0,q4_0,q4_1,q4_k,q2_k,nf4_0,q8_0_fp32,q8_0_fp16,q4_0_fp32,q4_0_fp16}
                        Scheme of LLM quantization mode. default is unset.
  -lqc LLM_QUANT_CONFIG, --llm-quant-config LLM_QUANT_CONFIG
                        Set the name of quantization config for advanced quantize.
  -lqr LLM_QUANT_RECIPE, --llm-quant-recipe LLM_QUANT_RECIPE
                        Set the name of quantization recipe for based quantize.
  -in INPUT_NAME, --input-name INPUT_NAME
                        Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;).
  -is INPUT_SHAPE, --input-shape INPUT_SHAPE
                        Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space.
  -on OUTPUT_NAME, --output-name OUTPUT_NAME
                        Set the name of output nodes. Multiple shapes are separated by semicolon(;).
  --model-format {keras,onnx,pb,tflite,pytorch,caffe}
                        Specify input model format:['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe']
  --reorder-pixel-format
                        If original model's input data pixel format is rgb, then covert it to bgr;otherwise, then convert it to rgb.
  --board {th1520,e907,c906,c908,c920,c920v2,rvm,x86_ref,c907,c907rv32,unset}
                        Set target device, default is unset.
  --hybrid-computing    Supports hybrid computing on multiple devices. Currently supports CPU and NPU for th1520.
  --link-lib {shl_pnna,shl_th1520,shl_c906,shl_c908,shl_c920,shl_rvm,unset}
                        Set link library for -D. default is unset
  -cd CALIBRATE_DATASET, --calibrate-dataset CALIBRATE_DATASET
                        Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt.
  -qs {int4_asym_w_sym,uint8_asym,int8_asym,int8_sym,int8_original,int8_asym_w_sym,int16_sym,float16,float16_w_int8,float32,unset}, 
            --quantization-scheme {int4_asym_w_sym,
                                   uint8_asym,
                                   int8_asym,
                                   int8_sym,
                                   int8_original,
                                   int8_asym_w_sym,
                                   int16_sym,
                                   float16,
                                   float16_w_int8,
                                   float32,
                                   unset}
                        Scheme of quantization. default is unset.
  --auto-hybrid-quantization
                        If set, quantize model automatically.
  --quantization-loss-algorithm {cos_similarity,mse,kl_divergence,cross_entropy,gini}
                        How to calculate accuracy loss for every layer.
  --quantization-loss-threshold QUANTIZATION_LOSS_THRESHOLD
                        The threshold that will determin thich layer will be quantized with hybrid way.If it is None, we will select threshold automatically.
  --dump-quantization-loss
                        If set, dump quantizaiton loss into file.
  --hybrid-quantization-scheme {int4_asym_w_sym,uint8_asym,int8_asym,int8_sym,int8_asym_w_sym,int16_sym,float16,float32,unset}
                        Scheme of hybrid quantization. default is unset.
  --hybrid-layer-name HYBRID_LAYER_NAME [HYBRID_LAYER_NAME ...]
                        Layer buffer name to use hybrid quantization.
  --low-bound-scale LOW_BOUND_SCALE
                        Enlarge the low bound of the data type of specified quantization mode to avoid overflow during calculation. It should be >=1
  --high-bound-scale HIGH_BOUND_SCALE
                        Reduce the high bound of the data type of specified quantization mode to avoid overflow during calculation. It should be in (0, 1]
  --fuse-conv-relu      Fuse the convolution and relu layer.
  --fuse-sigmoid-mul    Fuse the sigmoid and mul layer.
  --fuse-reshape-dense  Fuse the reshape and dense layer.
  --broadcast-quantization
                        Broadcast quantization parameters for special ops.
  --fuse-clip           Fuse clip's attr into pre layer's quantitative information. This flag is only valid when quantization is used.
  --fuse-zp2bias        Merge conv2d/dense zp to bias.
  --target-layout {NCHW,NHWC}
                        Set target layout.
  --output-layout {NCHW,NHWC}
                        Set output layout.
  --quantization-tool {default,ppq}
                        Model quantization tool, default is hhb.
  --lsq                 Whether uses lsq quantization algorithm.
  --lsq-lr LSQ_LR       Initial learning rate for lsq.
  --lsq-steps LSQ_STEPS
                        Training steps for lsq.
  --quant-device {cpu,cuda}
                        The device to quantize model, defaults to cpu.
  --cali-batch CALI_BATCH
                        The batch size for calibration.
  --align-elementwise {None,Align to Large}
                        The method that aligns the inputs quant config of elementwise ops.
  --matrix-extension-mlen MATRIX_EXTENSION_MLEN
                        Specify T-head Matrix extension's MLEN bit, default is 0, unuse matrix extension.
  -s DATA_SCALE, --data-scale DATA_SCALE
                        Scale number(mul) for inputs normalization(data=img*scale), default is 1.
  -sv DATA_SCALE_DIV, --data-scale-div DATA_SCALE_DIV
                        Scale number(div) for inputs normalization(data=img/scale), default is 1.
  -m DATA_MEAN, --data-mean DATA_MEAN
                        Set the mean value of input, multiple values are separated by space, default is 0.
  -r DATA_RESIZE, --data-resize DATA_RESIZE
                        Resize base size for input image to resize.
  --pixel-format {RGB,BGR}
                        The pixel format of image data, defalut is RGB
  --add-preprocess-node
                        Add preprocess node for model.
  --config-file CONFIG_FILE
                        Configue more complex parameters for executing the model.
  --generate-config     Generate config file
  -o OUTPUT, --output OUTPUT
                        The directory that holds the outputs.
  --trace {relay,qnn,csinn,csinn_acc} [{relay,qnn,csinn,csinn_acc} ...]
                        Generate unify tracing data.
  -sd SIMULATE_DATA, --simulate-data SIMULATE_DATA
                        Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt.
  --postprocess {top5,save,save_and_top5}
                        Set the mode of postprocess: 'top5' show top5 of output; 'save' save output to file;'save_and_top5' show top5 and save output to file. Default is top5
  --show-session-run-time
                        If set, print session run time.
  --model-save {run_only,save_only,save_and_run}
                        Whether save binary graph or run only. run_only: execute model only, not save binary graph. save_only: save binary graph only. save_and_run: execute and save model.
  --model-priority MODEL_PRIORITY
                        Set model priority, only for th1520 now. 0 is lowest, 1 is medium, 2 is highest.
  --without-preprocess  Do not generate preprocess codes.
  --input-memory-type {0,1,2} [{0,1,2} ...]
                        Set the memory type for input tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer.
  --output-memory-type {0,1,2} [{0,1,2} ...]
                        Set the memory type for output tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer.
  --memory-type {0,1,2}
                        Set the memory type for input and output tensors. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer.
  --ahead-of-time {intrinsic,unset}
                        AOT to generate intrinsic.
  --dynamic-shape       If set, don't quantize the model.
  --device-thread DEVICE_THREAD
                        Generate thread_num for target backend.

commands:
  {'simulate', 'import', 'codegen', 'profiler', 'quantize'}
    import              Import a model into relay ir
    codegen             Codegen the imported model
    profiler            profile model
    quantize            Quantize the imported model
    simulate            Simulate the imported model

 HHB Command Line Tools 
