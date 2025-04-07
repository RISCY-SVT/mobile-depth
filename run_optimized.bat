@echo off
echo Running with optimized settings...
set CUDA_LAUNCH_BLOCKING=1
python optimized_main.py --data_root ..\MidAir_dataset --datasets Kite_training --environments sunny --batch_size 12 --image_size 256 --num_workers 16 --prefetch_factor 4 --cache_size 38374 --mixed_precision --output_dir MidAir_optimized
pause
