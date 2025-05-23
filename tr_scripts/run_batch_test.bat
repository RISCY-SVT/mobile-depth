REM Description: Batch script to run the MobileDepth model on the MidAir dataset
@echo off
echo MobileDepth Model Testing Script
echo ==============================

set MODEL_PATH=MidAir_optimized\checkpoints\best_model.pth
set DATA_ROOT=..\MidAir_dataset
set OUTPUT_DIR=test_results

echo Setting CUDA memory handling environment variables
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=1

echo Running comparison test with ground truth (small dataset)...
python test_mobile_depth.py --model_path %MODEL_PATH% --data_root %DATA_ROOT% --dataset_type Kite_test --environments sunny --output_dir %OUTPUT_DIR%\sunny_comparison --with_gt_comparison --max_samples 50

echo Running test on cloudy environment (small dataset)...
python test_mobile_depth.py --model_path %MODEL_PATH% --data_root %DATA_ROOT% --dataset_type Kite_test --environments cloudy --output_dir %OUTPUT_DIR%\cloudy_test --max_samples 50

echo Running test on sunset environment (small dataset)...
python test_mobile_depth.py --model_path %MODEL_PATH% --data_root %DATA_ROOT% --dataset_type Kite_test --environments sunset --output_dir %OUTPUT_DIR%\sunset_test --max_samples 50

echo Running short test with PLE winter data (small dataset)...
python test_mobile_depth.py --model_path %MODEL_PATH% --data_root %DATA_ROOT% --dataset_type PLE_test --environments winter --output_dir %OUTPUT_DIR%\winter_test --max_samples 50

echo Test completed! Results saved to: %OUTPUT_DIR%
pause
