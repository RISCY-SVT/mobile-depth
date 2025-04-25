import os
import glob
import random
import time
import threading
import queue
import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import psutil
import gc
import logging
import weakref

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataLoader")

class RAMCachedDataset(Dataset):
    """
    Dataset с RAM-кешированием для быстрой загрузки данных на устройствах с большим объемом RAM.
    Оптимизирован для систем с большим объемом памяти, но медленным доступом к диску.
    """
    def __init__(self, root_dir, datasets, environments, image_size=320, is_train=True, 
                 val_split=0.1, depth_scale=100.0, max_total_images=None, 
                 max_imgs_per_trajectory=None, cache_size=10000, enable_cache=True,
                 enable_augmentation=True, debug=False):
        """
        Инициализация датасета с возможностью кеширования в RAM.
        
        Args:
            root_dir: Корневая директория датасета
            datasets: Список типов датасетов для включения (напр., ['Kite_training'])
            environments: Список сред для включения (напр., ['sunny', 'foggy'])
            image_size: Размер изображений после ресайза
            is_train: Является ли этот датасет тренировочным
            val_split: Доля данных для валидации
            depth_scale: Множитель для карт глубины
            max_total_images: Максимальное количество изображений (для быстрого тестирования)
            max_imgs_per_trajectory: Максимальное число изображений из каждой траектории
            cache_size: Максимальный размер кеша в RAM
            enable_cache: Включить ли кеширование в RAM
            enable_augmentation: Включить ли аугментацию данных
            debug: Включить отладочный вывод
        """
        self.root_dir = root_dir
        self.image_size = min(image_size, 640)  # Ограничиваем максимальный размер для экономии памяти
        self.depth_scale = depth_scale
        self.enable_cache = enable_cache
        self.enable_augmentation = enable_augmentation and is_train  # Аугментируем только тренировочные данные
        self.debug = debug
        self.is_train = is_train
        
        # Настройка RAM-кеша с использованием OrderedDict для LRU поведения
        self.cache_size = min(cache_size, 20000) if enable_cache else 0  # Ограничиваем макс. размер кеша
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cache_report_time = time.time()
        
        # Мониторинг использования памяти
        self.memory_usage = []
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # секунд
        
        # Инициализация путей к данным
        self.rgb_files = []
        self.depth_files = []
        
        # Статистика
        self.trajectories_processed = 0
        self.start_time = time.time()
        
        logger.info(f"Инициализация {'тренировочного' if is_train else 'валидационного'} датасета с RAM-кешированием")
        logger.info(f"  - Размер кеша: {self.cache_size} изображений")
        logger.info(f"  - Размер изображений: {image_size}x{image_size}")
        logger.info(f"  - Доступная системная RAM: {psutil.virtual_memory().available / (1024**3):.1f} ГБ")
        
        # Поиск всех пар изображений
        total_pairs_found = 0
        trajectories_with_limits = defaultdict(int)
        
        for dataset in datasets:
            for env in environments:
                env_path = os.path.join(root_dir, dataset, env)
                if not os.path.exists(env_path):
                    logger.info(f"Пропускаем {env_path}: директория не существует")
                    continue
                
                color_dir = os.path.join(env_path, 'color_left')
                depth_dir = os.path.join(env_path, 'depth')
                
                if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                    logger.info(f"Пропускаем {env_path}: отсутствуют директории color_left или depth")
                    continue
                
                logger.info(f"Обработка {dataset}/{env}")
                
                trajectory_pairs = 0
                for trajectory_dir in os.listdir(color_dir):
                    rgb_trajectory_path = os.path.join(color_dir, trajectory_dir)
                    depth_trajectory_path = os.path.join(depth_dir, trajectory_dir)
                    
                    if not os.path.isdir(rgb_trajectory_path) or not os.path.exists(depth_trajectory_path):
                        continue
                    
                    # Поиск изображений с разными возможными расширениями
                    rgb_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG', '*.png', '*.PNG']:
                        rgb_files.extend(glob.glob(os.path.join(rgb_trajectory_path, ext)))
                    
                    rgb_files = sorted(rgb_files)
                    
                    # Применяем ограничение по количеству изображений из траектории
                    if max_imgs_per_trajectory and len(rgb_files) > max_imgs_per_trajectory:
                        # Используем равномерно распределенные сэмплы по всей траектории
                        indices = np.linspace(0, len(rgb_files)-1, max_imgs_per_trajectory, dtype=int)
                        rgb_files = [rgb_files[i] for i in indices]
                        trajectories_with_limits[trajectory_dir] = len(indices)
                    
                    pairs_in_trajectory = 0
                    for rgb_file in rgb_files:
                        file_id = os.path.splitext(os.path.basename(rgb_file))[0]
                        
                        # Проверяем различные расширения для файлов глубины
                        depth_file = None
                        for ext in ['.png', '.PNG', '.exr', '.EXR']:
                            temp_path = os.path.join(depth_trajectory_path, f"{file_id}{ext}")
                            if os.path.exists(temp_path):
                                depth_file = temp_path
                                break
                        
                        if depth_file:
                            self.rgb_files.append(rgb_file)
                            self.depth_files.append(depth_file)
                            pairs_in_trajectory += 1
                    
                    if pairs_in_trajectory > 0:
                        trajectory_pairs += pairs_in_trajectory
                        self.trajectories_processed += 1
                        if self.debug:
                            logger.info(f"  Траектория {trajectory_dir}: {pairs_in_trajectory} пар")
                
                total_pairs_found += trajectory_pairs
                logger.info(f"  Найдено {trajectory_pairs} пар в {dataset}/{env}")
        
        if trajectories_with_limits:
            logger.info(f"Применены ограничения к {len(trajectories_with_limits)} траекториям")
        
        logger.info(f"Всего найдено пар: {total_pairs_found}")
        
        # Применяем глобальное ограничение на количество изображений если указано
        if max_total_images and total_pairs_found > max_total_images:
            random.seed(42)  # Для воспроизводимости
            indices = random.sample(range(total_pairs_found), max_total_images)
            self.rgb_files = [self.rgb_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]
            logger.info(f"Случайно выбрано {max_total_images} пар из {total_pairs_found}")
        
        # Разделение на train/val
        if total_pairs_found > 0:
            all_indices = list(range(len(self.rgb_files)))
            random.seed(42)  # Для воспроизводимости
            random.shuffle(all_indices)
            
            split_idx = int(val_split * len(all_indices))
            selected_indices = all_indices[split_idx:] if is_train else all_indices[:split_idx]
            
            self.rgb_files = [self.rgb_files[i] for i in selected_indices]
            self.depth_files = [self.depth_files[i] for i in selected_indices]
            
            logger.info(f"{'Тренировочных' if is_train else 'Валидационных'} примеров: {len(self.rgb_files)}")
        else:
            logger.warning("ВНИМАНИЕ: Не найдено подходящих пар изображений!")
        
        # Базовые трансформации (всегда применяются)
        self.rgb_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform_base = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Трансформации для аугментации данных (только для тренировки)
        if self.enable_augmentation:
            logger.info("Аугментация данных включена")
        
        # Инициализация очереди для предзагрузки
        self.prefetch_queue = queue.Queue(maxsize=50)  # Уменьшена максимальная очередь для уменьшения потребления памяти
        self.prefetch_idx = 0
        self.prefetch_running = False
        self.prefetch_thread = None
        
        # Запускаем предзагрузку если кеш включен
        if self.enable_cache and len(self.rgb_files) > 0:
            self._start_prefetching()
        
        # Регистрация финализатора для корректного завершения
        self._finalizer = weakref.finalize(self, self._cleanup)
    
    def _cleanup(self):
        """Гарантированная очистка ресурсов при удалении объекта"""
        self.shutdown()
    
    def _start_prefetching(self):
        """Запуск фонового потока для предзагрузки и кеширования данных"""
        if self.prefetch_running:
            return
            
        self.prefetch_running = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.info("Запущен фоновый поток предзагрузки")
    
    def _prefetch_worker(self):
        """Рабочий поток для предзагрузки данных в RAM-кеш с адаптивным управлением ресурсами"""
        logger.info("Поток предзагрузки запущен")
        try:
            indices = list(range(len(self.rgb_files)))
            random.shuffle(indices)  # Загружаем в случайном порядке для лучшего распределения кеша
            
            while self.prefetch_running:
                # Проверка использования памяти
                mem_percent = psutil.virtual_memory().percent
                
                # Если памяти осталось мало, прекращаем кеширование новых данных
                if mem_percent > 85:  # Критический уровень использования памяти
                    logger.warning(f"Высокое использование RAM ({mem_percent}%) - приостановка кеширования")
                    time.sleep(5)  # Ждем некоторое время перед следующей проверкой
                    continue
                
                # Проверяем размер кеша
                if len(self.cache) >= self.cache_size:
                    # Можем очистить некоторые старые элементы, если кеш заполнен
                    if len(self.cache) > self.cache_size + 100:
                        # Удаляем 10% самых старых элементов
                        to_remove = int(self.cache_size * 0.1)
                        for _ in range(to_remove):
                            if self.cache:
                                self.cache.popitem(last=False)
                    
                    # Ждем какое-то время перед следующей попыткой
                    time.sleep(0.5)
                    continue
                
                # Выбираем следующий индекс для кеширования
                for idx in indices:
                    if not self.prefetch_running:
                        break
                        
                    # Пропускаем если уже в кеше
                    if idx in self.cache:
                        continue
                    
                    # Загружаем данные
                    try:
                        rgb_path = self.rgb_files[idx]
                        depth_path = self.depth_files[idx]
                        
                        # Проверка доступного размера памяти
                        if psutil.virtual_memory().percent > 80:
                            logger.warning("Высокое использование RAM - пауза в кешировании")
                            break
                        
                        # Загрузка и предобработка RGB изображения
                        with Image.open(rgb_path) as rgb_image:
                            rgb_image = rgb_image.convert('RGB')
                            rgb_tensor = self.rgb_transform_base(rgb_image)
                        
                        # Загрузка и предобработка изображения глубины
                        with Image.open(depth_path) as depth_image:
                            depth_tensor = self.depth_transform_base(depth_image)
                        
                        # Нормализация глубины
                        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
                        if depth_max > depth_min:  # Проверка для избежания деления на ноль
                            depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
                        
                        # Защита от экстремальных значений
                        depth_tensor = torch.clamp(depth_tensor, 0.001, 0.999)
                        
                        # Обеспечиваем одноканальную глубину
                        if depth_tensor.shape[0] > 1:
                            depth_tensor = depth_tensor[0].unsqueeze(0)
                        
                        # Сохраняем в кеше
                        self.cache[idx] = (rgb_tensor, depth_tensor)
                        
                        # Отслеживаем использование памяти периодически
                        current_time = time.time()
                        if current_time - self.last_memory_check > self.memory_check_interval:
                            mem_usage = psutil.virtual_memory().percent
                            self.memory_usage.append(mem_usage)
                            self.last_memory_check = current_time
                            
                            # Отчет о статистике кеша периодически
                            if current_time - self.last_cache_report_time > 60:  # Отчет каждую минуту
                                self._report_cache_stats()
                                self.last_cache_report_time = current_time
                            
                            # Защита от исчерпания памяти
                            if mem_usage > 90:
                                logger.warning(f"Критическое использование RAM: {mem_usage}% - очистка кеша")
                                # Очищаем половину кеша
                                items_to_clear = len(self.cache) // 2
                                for _ in range(items_to_clear):
                                    if self.cache:
                                        self.cache.popitem(last=False)
                                # Форсируем сборку мусора
                                gc.collect()
                                break
                                
                    except Exception as e:
                        logger.error(f"Ошибка при кешировании индекса {idx}: {e}")
                    
                    # Освобождаем немного времени CPU
                    time.sleep(0.001)
                
                # Перемешиваем индексы после полного прохода
                random.shuffle(indices)
        
        except Exception as e:
            logger.error(f"Исключение в потоке предзагрузки: {e}")
        
        finally:
            self.prefetch_running = False
            logger.info("Поток предзагрузки остановлен")
    
    def _report_cache_stats(self):
        """Отчет о статистике кеша"""
        try:
            cache_size_mb = sum(t.element_size() * t.nelement() for pair in self.cache.values() for t in pair) / (1024**2)
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / max(1, total_requests) * 100
            
            logger.info(f"Статистика кеша: {len(self.cache)}/{self.cache_size} элементов, {cache_size_mb:.1f} МБ, "
                      f"Попадания: {hit_rate:.1f}% ({self.cache_hits}/{total_requests})")
            logger.info(f"Использование памяти: {psutil.virtual_memory().percent}%, "
                      f"Доступно: {psutil.virtual_memory().available / (1024**3):.1f} ГБ")
            
            # Сброс счетчиков
            self.cache_hits = 0
            self.cache_misses = 0
        except Exception as e:
            logger.error(f"Ошибка при создании отчета о кеше: {e}")
    
    def _apply_augmentation(self, rgb, depth):
        """Применение аугментации к паре RGB и depth"""
        if not self.enable_augmentation:
            return rgb, depth
        
        # Случайное отражение по горизонтали (50% вероятность)
        if random.random() > 0.5:
            rgb = torch.flip(rgb, [2])  # Отражение по горизонтали
            depth = torch.flip(depth, [2])  # Отражение по горизонтали
        
        # Случайное изменение яркости и контраста RGB (30% вероятность)
        if random.random() > 0.7:
            try:
                # Применяем изменения к RGB
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                
                # Денормализация
                means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                rgb_denorm = rgb * stds + means
                
                # Изменение яркости и контраста
                rgb_denorm = torch.clamp(rgb_denorm * brightness_factor, 0, 1)
                gray = rgb_denorm.mean(dim=0, keepdim=True)
                rgb_denorm = torch.clamp((rgb_denorm - gray) * contrast_factor + gray, 0, 1)
                
                # Ренормализация
                rgb = (rgb_denorm - means) / stds
            except Exception as e:
                logger.warning(f"Ошибка при аугментации: {e}")
        
        return rgb, depth
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        """Получение элемента с оптимизацией кеширования и управления памятью"""
        # Проверка наличия в кеше
        if self.enable_cache and idx in self.cache:
            self.cache_hits += 1
            
            # Перемещаем элемент в конец OrderedDict (недавно использованный)
            try:
                rgb_tensor, depth_tensor = self.cache[idx]
                # Обновляем позицию в кеше (LRU стратегия)
                self.cache.pop(idx)
                self.cache[idx] = (rgb_tensor, depth_tensor)
                
                # Применяем аугментацию
                if self.enable_augmentation:
                    rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
                    
                return {
                    'rgb': rgb_tensor,
                    'depth': depth_tensor,
                    'rgb_path': self.rgb_files[idx],
                    'depth_path': self.depth_files[idx],
                    'from_cache': True
                }
            except Exception as e:
                logger.warning(f"Ошибка при получении из кеша индекса {idx}: {e}")
                # Продолжаем, как будто это был промах в кеше
                self.cache_misses += 1
        else:
            self.cache_misses += 1
        
        # Промах в кеше - загрузка с диска
        try:
            # Загрузка RGB изображения
            rgb_path = self.rgb_files[idx]
            with Image.open(rgb_path) as rgb_image:
                rgb_image = rgb_image.convert('RGB')
                rgb_tensor = self.rgb_transform_base(rgb_image)
            
            # Загрузка изображения глубины
            depth_path = self.depth_files[idx]
            with Image.open(depth_path) as depth_image:
                depth_tensor = self.depth_transform_base(depth_image)
            
            # Нормализация глубины с защитой от нулевых значений
            depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
            if depth_max > depth_min:  # Проверка для избежания деления на ноль
                depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Защита от экстремальных значений для лучшей квантизации
            depth_tensor = torch.clamp(depth_tensor, 0.001, 0.999)
            
            # Обеспечиваем одноканальную глубину
            if depth_tensor.shape[0] > 1:
                depth_tensor = depth_tensor[0].unsqueeze(0)
            
            # Применяем аугментацию если включена
            if self.enable_augmentation:
                rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
            
            # Добавляем в кеш если включен и память не перегружена
            if self.enable_cache and psutil.virtual_memory().percent < 85:
                # Если кеш полон, удаляем самый старый элемент (первый в OrderedDict)
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                
                # Добавляем текущий элемент в кеш
                self.cache[idx] = (rgb_tensor.clone(), depth_tensor.clone())
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'from_cache': False
            }
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для индекса {idx}: {e}")
            
            # Возвращаем заглушки нулевых тензоров в случае ошибки
            rgb_tensor = torch.zeros(3, self.image_size, self.image_size)
            depth_tensor = torch.zeros(1, self.image_size, self.image_size)
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': self.rgb_files[idx] if idx < len(self.rgb_files) else "unknown",
                'depth_path': self.depth_files[idx] if idx < len(self.depth_files) else "unknown",
                'from_cache': False,
                'error': True
            }
    
    def shutdown(self):
        """Корректное завершение работы потока предзагрузки"""
        logger.info("Завершение работы датасета...")
        self.prefetch_running = False
        
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
            logger.info("Поток предзагрузки успешно остановлен")
            
        self._report_cache_stats()
        
        # Очистка кеша для освобождения памяти
        self.cache.clear()
        gc.collect()
        logger.info("Датасет корректно завершил работу")
