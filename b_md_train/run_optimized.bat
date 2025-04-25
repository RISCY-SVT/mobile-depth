@echo off
echo Running with B_MobileDepth optimized settings...

:: Оптимизация CUDA
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: Размеры батча и потоков
set BATCH_SIZE=8
set NUM_WORKERS=4
set PREFETCH_FACTOR=2
set EPOCHS=10

:: Размер изображения 
set IMAGE_SIZE=320

python optimized_main.py ^
  --data_root ..\MidAir_dataset ^
  --datasets Kite_training ^
  --environments sunny ^
  --batch_size %BATCH_SIZE% ^
  --image_size %IMAGE_SIZE% ^
  --num_workers %NUM_WORKERS% ^
  --prefetch_factor %PREFETCH_FACTOR% ^
  --cache_size 50000 ^
  --mixed_precision ^
  --output_dir B_MobileDepth_trained ^
  --lr 0.0002 ^
  --weight_decay 1e-4 ^
  --clip_grad 1.0 ^
  --min_lr 1e-6 ^
  --epochs %EPOCHS% ^
  --use_robust_loss ^
  --test_quantization

echo Training complete! Press any key to exit.

rem  --strong_augmentation ^
rem  --noise_augmentation ^
rem  --perspective_augmentation ^
