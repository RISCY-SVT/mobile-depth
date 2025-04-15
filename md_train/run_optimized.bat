@echo off
echo Running with optimized settings...

:: Оптимизация CUDA
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: Размеры батча и потоков
set BATCH_SIZE=12
set NUM_WORKERS=8
set PREFETCH_FACTOR=4
set EPOCHS=10

:: Размер изображения 
set IMAGE_SIZE=256

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
  --output_dir MidAir_enhanced ^
  --lr 0.0002 ^
  --weight_decay 1e-4 ^
  --clip_grad 1.0 ^
  --min_lr 1e-6 ^
  --epochs %EPOCHS% ^
  --resume .\MidAir_enhanced\checkpoints\model_epoch_9.pth

pause
