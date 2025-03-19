@echo off
echo Running with optimized settings...
python optimized_main.py --data_root ..\MidAir_dataset --datasets Kite_training --environments sunny --batch_size 12 --image_size 384 --num_workers 16 --prefetch_factor 4 --cache_size 38374 --mixed_precision --output_dir MidAir_optimized 
pause
