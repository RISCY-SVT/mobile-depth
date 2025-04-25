import matplotlib
matplotlib.use('Agg')  # Использование неинтерактивного бэкенда
import os
import argparse
import time
import torch
from bdepth_model import B_MobileDepth

# Импорт оптимизированных компонентов
from improved_fix_data_loader import EnhancedRAMCachedDataset, create_improved_dataloader
from improved_training_loop import train_improved

import logging
import gc
import sys
import psutil
import numpy as np

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger("OptimizedMain")

# Настройка CUDA и выделения памяти для GTX1080
def setup_gpu_memory():
    """Настройка оптимального использования GPU памяти для GTX 1080"""
    if torch.cuda.is_available():
        # Отключаем асинхронные CUDA операции для более предсказуемого использования памяти
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False  
        torch.backends.cudnn.deterministic = True
        
        # Ограничение использования памяти до 85% от доступной для избежания OOM
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = int(total_memory * 0.85)
        
        # Для PyTorch 1.10+
        if hasattr(torch.cuda, 'memory_reserved'):
            logger.info(f"Настройка максимальной CUDA памяти: {max_memory / (1024**3):.2f} ГБ")
            # Управление фрагментацией памяти
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            torch.cuda.empty_cache()
        
        # Отключаем кэширование выделений CUDA для более явного контроля памяти
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # Вывод информации о GPU
        logger.info(f"Используем GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Общая память GPU: {total_memory / (1024**3):.2f} ГБ")
        logger.info(f"Выделенная память GPU: {torch.cuda.memory_allocated() / (1024**3):.4f} ГБ")
        logger.info(f"Зарезервированная память GPU: {torch.cuda.memory_reserved() / (1024**3):.4f} ГБ")
    else:
        logger.warning("CUDA недоступна, используем CPU!")

def main():
    # Применяем оптимальные настройки для GPU памяти
    setup_gpu_memory()
    
    # Очистка CUDA кеша перед началом работы
    torch.cuda.empty_cache()
    gc.collect()
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Оптимизированное обучение B_MobileDepth модели на MidAir датасете')
    
    # Параметры датасета
    parser.add_argument('--data_root', type=str, default='MidAir_dataset', help='Корневая директория датасета')
    parser.add_argument('--datasets', type=str, nargs='+', default=['Kite_training'], 
                      help='Типы датасетов (Kite_training, PLE_training)')
    parser.add_argument('--environments', type=str, nargs='+', default=['sunny', 'foggy', 'cloudy', 'sunset'], 
                      help='Окружающие среды (sunny, foggy, cloudy, sunset, fall, spring, winter)')
    parser.add_argument('--max_total_images', type=int, default=None, 
                      help='Максимальное общее количество изображений (None = использовать все)')
    parser.add_argument('--max_imgs_per_trajectory', type=int, default=None, 
                      help='Максимальное количество изображений на траекторию (None = использовать все)')
    parser.add_argument('--image_size', type=int, default=256, 
                      help='Размер изображения (меньше = быстрее обучение)')
    parser.add_argument('--depth_scale', type=float, default=100.0, 
                      help='Масштабный коэффициент для карт глубины')
    parser.add_argument('--val_split', type=float, default=0.1, 
                      help='Доля данных для валидации')
    
    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=8, 
                      help='Размер батча для обучения (для GTX1080 рекомендуется 4-8)')
    parser.add_argument('--epochs', type=int, default=15, 
                      help='Количество эпох для обучения')
    parser.add_argument('--lr', type=float, default=0.0002, 
                      help='Начальная скорость обучения')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                      help='Весовое затухание для оптимизатора')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                      help='Значение для ограничения градиентов')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                      help='Минимальная скорость обучения')
    parser.add_argument('--val_freq', type=int, default=1, 
                      help='Проверять на валидационном наборе каждые N эпох')
    parser.add_argument('--save_freq', type=int, default=1, 
                      help='Сохранять контрольную точку каждые N эпох')
    parser.add_argument('--resume', type=str, default=None, 
                      help='Путь к контрольной точке для продолжения обучения')
    
    # Параметры вывода
    parser.add_argument('--output_dir', type=str, default='output', 
                      help='Директория для сохранения результатов')
    
    # Параметры производительности
    parser.add_argument('--mixed_precision', action='store_true', 
                      help='Использовать смешанную точность для обучения')
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Количество рабочих процессов для загрузки данных')
    parser.add_argument('--prefetch_factor', type=int, default=2, 
                      help='Количество предварительно загружаемых батчей на одного рабочего')
    parser.add_argument('--cache_size', type=int, default=10000, 
                      help='Количество изображений для кеширования в RAM')
    parser.add_argument('--pin_memory', action='store_true', default=True, 
                      help='Прикреплять память для более быстрой передачи на GPU')
    parser.add_argument('--disable_cache', action='store_true', 
                      help='Отключить кеширование в RAM')
    parser.add_argument('--disable_augmentation', action='store_true', 
                      help='Отключить аугментацию данных')
    
    # Параметры аугментации для B_MobileDepth
    parser.add_argument('--strong_augmentation', action='store_true', 
                      help='Включить более сильную аугментацию данных')
    parser.add_argument('--noise_augmentation', action='store_true', 
                      help='Добавить шумовую аугментацию для лучшей квантизации')
    parser.add_argument('--perspective_augmentation', action='store_true', 
                      help='Добавить перспективные трансформации для аугментации')
    
    # Параметры квантизации
    parser.add_argument('--use_qat', action='store_true', 
                      help='Использовать Quantization-Aware Training')
    parser.add_argument('--test_quantization', action='store_true', 
                      help='Протестировать модель с симулированной квантизацией')
    parser.add_argument('--use_robust_loss', action='store_true', 
                      help='Использовать более устойчивую функцию потерь')
    
    # Параметры отладки
    parser.add_argument('--debug', action='store_true', 
                      help='Включить отладочный вывод')
    parser.add_argument('--profile', action='store_true', 
                      help='Включить профилирование')
    
    args = parser.parse_args()
    
    # Печать информации о системе
    logger.info("\n=== Информация о системе ===")
    logger.info(f"Версия PyTorch: {torch.__version__}")
    logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Версия CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} ГБ")
    
    logger.info(f"CPU: {psutil.cpu_count(logical=True)} логических ядер")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} ГБ")
    logger.info(f"Доступно RAM: {psutil.virtual_memory().available / (1024**3):.2f} ГБ")
    
    # Измерение времени выполнения скрипта
    script_start_time = time.time()
    
    # Печать настроек оптимизации
    logger.info("\n=== Настройки оптимизации ===")
    logger.info(f"Смешанная точность: {args.mixed_precision}")
    logger.info(f"Количество рабочих: {args.num_workers}")
    logger.info(f"Размер батча: {args.batch_size}")
    logger.info(f"Размер изображения: {args.image_size}")
    logger.info(f"RAM кеш: {'Отключен' if args.disable_cache else f'Включен, размер: {args.cache_size}'}")
    logger.info(f"Аугментация данных: {'Отключена' if args.disable_augmentation else 'Включена'}")
    logger.info(f"Сильная аугментация: {'Включена' if args.strong_augmentation else 'Отключена'}")
    logger.info(f"Шумовая аугментация: {'Включена' if args.noise_augmentation else 'Отключена'}")
    logger.info(f"Перспективная аугментация: {'Включена' if args.perspective_augmentation else 'Отключена'}")
    logger.info(f"Quantization-Aware Training: {'Включен' if args.use_qat else 'Отключен'}")
    logger.info(f"Устойчивая функция потерь: {'Включена' if args.use_robust_loss else 'Отключена'}")
    
    # Создание директории для вывода
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Инициализация датасетов
    logger.info("\n=== Загрузка датасетов ===")
    
    # Адаптивно уменьшаем размер батча для GTX 1080, если указан большой
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem_gb <= 8.5:  # GTX 1080 или аналогичный с 8ГБ
            logger.info("Обнаружена видеокарта с 8ГБ VRAM - применяются оптимизированные настройки")
            
            # Уменьшаем размер батча если был указан слишком большой
            if args.batch_size > 8 and args.image_size >= 320:
                original_batch = args.batch_size
                args.batch_size = 8
                logger.warning(f"Размер батча уменьшен с {original_batch} до {args.batch_size} "
                             f"для работы на GTX 1080 с изображениями {args.image_size}x{args.image_size}")
            elif args.batch_size > 4 and args.image_size >= 384:
                original_batch = args.batch_size
                args.batch_size = 4
                logger.warning(f"Размер батча уменьшен с {original_batch} до {args.batch_size} "
                             f"для работы на GTX 1080 с изображениями {args.image_size}x{args.image_size}")
    
    # Инициализируем тренировочный датасет с улучшенным кешированием
    try:
        train_dataset = EnhancedRAMCachedDataset(
            root_dir=args.data_root,
            datasets=args.datasets,
            environments=args.environments,
            image_size=args.image_size,
            is_train=True,
            val_split=args.val_split,
            depth_scale=args.depth_scale,
            max_total_images=args.max_total_images,
            max_imgs_per_trajectory=args.max_imgs_per_trajectory,
            cache_size=args.cache_size if not args.disable_cache else 0,
            enable_cache=not args.disable_cache,
            enable_augmentation=not args.disable_augmentation,
            strong_augmentation=args.strong_augmentation,
            noise_augmentation=args.noise_augmentation,
            perspective_augmentation=args.perspective_augmentation,
            debug=args.debug
        )

        # Инициализируем валидационный датасет
        val_dataset = EnhancedRAMCachedDataset(
            root_dir=args.data_root,
            datasets=args.datasets,
            environments=args.environments,
            image_size=args.image_size,
            is_train=False,
            val_split=args.val_split,
            depth_scale=args.depth_scale,
            max_total_images=None,  # Используем все валидационные данные
            max_imgs_per_trajectory=args.max_imgs_per_trajectory,
            cache_size=min(args.cache_size // 5, 5000) if not args.disable_cache else 0,  # Меньший кеш для валидации
            enable_cache=not args.disable_cache,
            enable_augmentation=False,  # Нет аугментации для валидации
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Ошибка при создании датасетов: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Пробуем использовать запасной вариант датасета...")
        
        # Запасной вариант с обычным датасетом
        from optimized_data_loader import RAMCachedDataset
        train_dataset = RAMCachedDataset(
            root_dir=args.data_root,
            datasets=args.datasets,
            environments=args.environments,
            image_size=args.image_size,
            is_train=True,
            val_split=args.val_split,
            depth_scale=args.depth_scale,
            max_total_images=args.max_total_images,
            max_imgs_per_trajectory=args.max_imgs_per_trajectory,
            cache_size=args.cache_size if not args.disable_cache else 0,
            enable_cache=not args.disable_cache,
            enable_augmentation=not args.disable_augmentation,
            debug=args.debug
        )
        
        val_dataset = RAMCachedDataset(
            root_dir=args.data_root,
            datasets=args.datasets,
            environments=args.environments,
            image_size=args.image_size,
            is_train=False,
            val_split=args.val_split,
            depth_scale=args.depth_scale,
            max_total_images=None,
            max_imgs_per_trajectory=args.max_imgs_per_trajectory,
            cache_size=min(args.cache_size // 5, 5000) if not args.disable_cache else 0,
            enable_cache=not args.disable_cache,
            enable_augmentation=False,
            debug=args.debug
        )
    
    # Создание модели
    logger.info("\n=== Инициализация модели B_MobileDepth ===")
    try:
        model = B_MobileDepth(input_size=(args.image_size, args.image_size))
        
        # Получение информации о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Параметры модели: {total_params:,} всего, {trainable_params:,} обучаемых")
        
        # Проверка требуемой памяти модели с учетом батчей и вычислений
        batch_size = args.batch_size
        input_size = args.image_size
        
        # Примерная оценка требуемой памяти GPU (грубое приближение)
        if torch.cuda.is_available():
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            # Буферы для optimizers, gradients, activations и т.д.
            estimated_batch_memory = batch_size * input_size * input_size * 4 * 12 / (1024**3)  # Примерно 12 тензоров размера (B,C,H,W)
            total_estimated = param_memory + estimated_batch_memory
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_usage_percent = (total_estimated / gpu_memory) * 100
            
            logger.info(f"Оценка памяти модели: {param_memory:.2f} ГБ")
            logger.info(f"Оценка памяти батча: {estimated_batch_memory:.2f} ГБ")
            logger.info(f"Общая оценка: {total_estimated:.2f} ГБ ({memory_usage_percent:.1f}% от {gpu_memory:.1f} ГБ)")
            
            # Предупреждение если модель слишком велика для имеющейся памяти
            if memory_usage_percent > 70:
                logger.warning(f"ВНИМАНИЕ: Модель может использовать более 70% доступной GPU памяти!")
                logger.warning(f"Рекомендации: уменьшить batch_size или image_size")
        
        # Профилирование по запросу
        if args.profile:
            try:
                from torch.profiler import profile, record_function, ProfilerActivity
                logger.info("Профилирование включено - будет создан отчет о профилировании после первой эпохи")
            except ImportError:
                logger.info("Профилирование запрошено, но torch.profiler недоступен")
                args.profile = False
        
        # Обучение модели
        logger.info("\n=== Начало обучения ===")
        model = train_improved(args, model, train_dataset, val_dataset)
        
        # Финальная очистка и измерение времени
        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"\n=== Обучение завершено ===")
        logger.info(f"Общее время выполнения: {int(hours)}ч {int(minutes)}м {int(seconds)}с")
        logger.info(f"Результаты сохранены в: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка в основном процессе обучения: {e}")
        import traceback
        traceback.print_exc()
        
        # Аварийное сохранение модели если возможно
        try:
            if 'model' in locals():
                emergency_path = os.path.join(args.output_dir, 'emergency_model.pth')
                torch.save(model.state_dict(), emergency_path)
                logger.info(f"Аварийное сохранение модели в {emergency_path}")
        except:
            logger.error("Не удалось сохранить модель в аварийном режиме")
        
        # Очистка ресурсов
        if 'train_dataset' in locals() and hasattr(train_dataset, 'shutdown'):
            train_dataset.shutdown()
        if 'val_dataset' in locals() and hasattr(val_dataset, 'shutdown'):
            val_dataset.shutdown()
        
        # Освобождение CUDA памяти
        torch.cuda.empty_cache()
        gc.collect()
        
        return 1  # Код ошибки
    
    return 0  # Успешное завершение

if __name__ == "__main__":
    try:
        # Установка ограничений на использование памяти
        if torch.cuda.is_available():
            # Настройка для GTX 1080 и других видеокарт с 8ГБ памяти
            torch.cuda.empty_cache()
            
            # Установка ограничений использования GPU памяти
            if torch.cuda.get_device_properties(0).total_memory / (1024**3) <= 8.5:
                # Это похоже на GTX 1080 или другую карту с 8ГБ VRAM
                torch.cuda.set_per_process_memory_fraction(0.85)  # Оставляем запас в 15%
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                logger.info("Применены ограничения памяти для GTX 1080")
        
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Необработанное исключение: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
