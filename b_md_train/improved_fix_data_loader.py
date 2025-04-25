import os
import torch
import random
import numpy as np
import threading
import time
import psutil
import gc
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Настройка размеров для защиты от OOM на GTX 1080
MAX_SAFE_IMAGE_SIZE = 320
MAX_SAFE_CACHE_ENTRIES = 2000  # Уменьшенный кеш для уменьшения нагрузки на RAM

def seed_worker(worker_id):
    """Инициализация seed для каждого worker'а чтобы гарантировать воспроизводимость"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Гарантируем, что мы не перегружаем системную память
    gc.collect()

class EnhancedRAMCachedDataset(Dataset):
    """
    Улучшенный кеширующий датасет с поддержкой продвинутых аугментаций,
    оптимизированный для работы с ограниченной памятью GPU.
    """
    def __init__(self, 
                 root_dir, 
                 datasets, 
                 environments, 
                 image_size=320, 
                 is_train=True, 
                 val_split=0.1, 
                 depth_scale=100.0, 
                 max_total_images=None, 
                 max_imgs_per_trajectory=None, 
                 cache_size=2000,  # Уменьшенный размер кеша по умолчанию 
                 enable_cache=True,
                 enable_augmentation=True, 
                 strong_augmentation=False,
                 noise_augmentation=False,
                 perspective_augmentation=False,
                 normalize_depth=True,
                 debug=False):
        """
        Инициализация улучшенного датасета.
        
        Аргументы:
            root_dir: Корневая директория данных
            datasets: Список типов датасетов для включения
            environments: Список окружений для включения
            image_size: Размер для ресайза изображений
            is_train: Флаг, указывающий на тренировочный датасет
            val_split: Доля данных для валидационного сплита
            depth_scale: Множитель для масштабирования карт глубины
            max_total_images: Максимальное количество изображений (для теста)
            max_imgs_per_trajectory: Максимальное число кадров из траектории
            cache_size: Максимальный размер кеша в RAM (количество изображений)
            enable_cache: Включить кеширование в RAM
            enable_augmentation: Включить аугментацию данных
            strong_augmentation: Использовать усиленную аугментацию
            noise_augmentation: Добавлять шум (для устойчивости к квантизации)
            perspective_augmentation: Применять перспективные искажения
            normalize_depth: Нормализовать карты глубины
            debug: Режим отладки
        """
        # Основные параметры
        self.root_dir = root_dir
        self.image_size = min(image_size, MAX_SAFE_IMAGE_SIZE)  # Ограничиваем размер для GTX 1080
        self.depth_scale = depth_scale
        self.enable_augmentation = enable_augmentation and is_train  # Аугментация только для тренировки
        self.normalize_depth = normalize_depth
        self.debug = debug
        self.is_train = is_train
        
        # Параметры аугментации
        self.strong_augmentation = strong_augmentation
        self.noise_augmentation = noise_augmentation
        self.perspective_augmentation = perspective_augmentation
        
        # Параметры кеширования
        self.enable_cache = enable_cache
        self.cache_size = min(cache_size, MAX_SAFE_CACHE_ENTRIES) if enable_cache else 0
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Пути к изображениям
        self.rgb_files = []
        self.depth_files = []
        
        print(f"Инициализация {'тренировочного' if is_train else 'валидационного'} датасета")
        print(f"  - Размер кеша: {self.cache_size} изображений")
        print(f"  - Размер изображений: {self.image_size}x{self.image_size}")
        print(f"  - Доступная RAM: {psutil.virtual_memory().available / (1024**3):.1f} ГБ")
        print(f"  - Аугментация: {'Включена' if self.enable_augmentation else 'Отключена'}")
        
        # Поиск пар изображений (RGB + depth)
        total_pairs_found = 0
        
        for dataset in datasets:
            for env in environments:
                env_path = os.path.join(root_dir, dataset, env)
                if not os.path.exists(env_path):
                    if debug:
                        print(f"Пропускаем {env_path}: директория не существует")
                    continue
                
                color_dir = os.path.join(env_path, 'color_left')
                depth_dir = os.path.join(env_path, 'depth')
                
                if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                    if debug:
                        print(f"Пропускаем {env_path}: отсутствуют директории color_left или depth")
                    continue
                
                print(f"Обработка {dataset}/{env}")
                
                trajectory_pairs = self._process_trajectories(
                    color_dir, depth_dir, max_imgs_per_trajectory, dataset, env
                )
                
                total_pairs_found += trajectory_pairs
                print(f"  Найдено {trajectory_pairs} пар в {dataset}/{env}")
        
        print(f"Всего найдено пар: {total_pairs_found}")
        
        # Ограничение общего числа изображений
        if max_total_images and total_pairs_found > max_total_images:
            random.seed(42)  # Для воспроизводимости
            indices = random.sample(range(total_pairs_found), max_total_images)
            self.rgb_files = [self.rgb_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]
            print(f"Выбрано случайно {max_total_images} пар из {total_pairs_found}")
        
        # Разделение на train/val
        if total_pairs_found > 0:
            all_indices = list(range(len(self.rgb_files)))
            random.seed(42)  # Для воспроизводимости
            random.shuffle(all_indices)
            
            split_idx = int(val_split * len(all_indices))
            selected_indices = all_indices[split_idx:] if is_train else all_indices[:split_idx]
            
            self.rgb_files = [self.rgb_files[i] for i in selected_indices]
            self.depth_files = [self.depth_files[i] for i in selected_indices]
            
            print(f"{'Тренировочных' if is_train else 'Валидационных'} примеров: {len(self.rgb_files)}")
        else:
            print("ВНИМАНИЕ: Не найдено подходящих пар изображений!")
        
        # Базовые трансформации (применяются всегда)
        self.rgb_transform_base = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform_base = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Дополнительные трансформации аугментации
        if self.enable_augmentation:
            self._setup_enhanced_augmentations()
    
    def _process_trajectories(self, color_dir, depth_dir, max_imgs_per_trajectory, dataset, env):
        """Обрабатывает траектории и находит пары изображений (RGB + depth)"""
        trajectory_pairs = 0
        
        for trajectory_dir in os.listdir(color_dir):
            rgb_trajectory_path = os.path.join(color_dir, trajectory_dir)
            depth_trajectory_path = os.path.join(depth_dir, trajectory_dir)
            
            if not os.path.isdir(rgb_trajectory_path) or not os.path.exists(depth_trajectory_path):
                continue
            
            # Поиск изображений с разными возможными расширениями
            rgb_files = []
            for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG', '*.png', '*.PNG']:
                rgb_files.extend(
                    [f for f in os.listdir(rgb_trajectory_path) 
                     if f.lower().endswith(ext.replace('*', ''))]
                )
            
            rgb_files = sorted(rgb_files)
            
            # Применение ограничения количества изображений из траектории
            if max_imgs_per_trajectory and len(rgb_files) > max_imgs_per_trajectory:
                # Выбираем равномерно распределенные кадры
                indices = np.linspace(0, len(rgb_files)-1, max_imgs_per_trajectory, dtype=int)
                rgb_files = [rgb_files[i] for i in indices]
            
            # Находим пары RGB + depth
            pairs_in_trajectory = 0
            for rgb_file in rgb_files:
                file_id = os.path.splitext(os.path.basename(rgb_file))[0]
                
                # Проверяем разные возможные расширения для depth
                depth_file = None
                for ext in ['.png', '.PNG', '.exr', '.EXR']:
                    temp_depth_file = os.path.join(depth_trajectory_path, f"{file_id}{ext}")
                    if os.path.exists(temp_depth_file):
                        depth_file = temp_depth_file
                        break
                
                if depth_file:
                    self.rgb_files.append(os.path.join(rgb_trajectory_path, rgb_file))
                    self.depth_files.append(depth_file)
                    pairs_in_trajectory += 1
            
            trajectory_pairs += pairs_in_trajectory
            
            if pairs_in_trajectory == 0 and self.debug:
                print(f"  Внимание: не найдено пар в траектории {dataset}/{env}/{trajectory_dir}")
        
        return trajectory_pairs
    
    def _setup_enhanced_augmentations(self):
        """Настройка улучшенных аугментаций для лучшей генерализации и устойчивости к квантизации"""
        brightness = 0.2 if self.strong_augmentation else 0.1
        contrast = 0.2 if self.strong_augmentation else 0.1
        saturation = 0.2 if self.strong_augmentation else 0.1
        hue = 0.1 if self.strong_augmentation else 0.05
        
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
        # Перспективные и аффинные трансформации
        if self.perspective_augmentation:
            distortion_scale = 0.1 if self.strong_augmentation else 0.05
            self.perspective_transform = transforms.RandomPerspective(
                distortion_scale=distortion_scale, 
                p=0.3
            )
            
            self.affine_transform = transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=2
            )
    
    def _apply_augmentation(self, rgb, depth):
        """Применяет аугментацию к паре RGB и depth с улучшенной устойчивостью к квантизации"""
        if not self.enable_augmentation:
            return rgb, depth
        
        # Устанавливаем одинаковый seed для согласованности между RGB и depth
        seed = torch.randint(0, 2**31, (1,)).item()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Случайное отражение по горизонтали (50% вероятность)
        if random.random() > 0.5:
            rgb = torch.flip(rgb, [2])  # Отражение по горизонтали
            depth = torch.flip(depth, [2])  # Отражение по горизонтали
        
        # Продвинутые цветовые аугментации для RGB изображения
        if random.random() > 0.5:
            try:
                # Преобразуем для цветовой коррекции
                rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
                rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_np)
                
                # Применяем цветовую коррекцию
                rgb_pil = self.color_jitter(rgb_pil)
                
                # Конвертируем обратно в тензор
                rgb = transforms.ToTensor()(rgb_pil)
                
                # Ренормализуем
                rgb = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(rgb)
            except Exception as e:
                print(f"Ошибка при цветовой аугментации: {e}")
        
        # Перспективные и аффинные трансформации
        if self.perspective_augmentation and random.random() > 0.7:
            try:
                # Преобразуем изображения для трансформаций
                rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
                rgb_np = np.clip(rgb_np, 0, 1)
                rgb_pil = Image.fromarray((rgb_np * 255).astype(np.uint8))
                
                depth_np = depth.permute(1, 2, 0).cpu().numpy()
                depth_np = np.clip(depth_np, 0, 1)
                depth_pil = Image.fromarray((depth_np * 255).astype(np.uint8))
                
                # Применяем одинаковую трансформацию к обоим изображениям
                if random.random() > 0.5:
                    # Перспективная трансформация
                    seed = random.randint(0, 2**31)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    rgb_pil = self.perspective_transform(rgb_pil)
                    
                    random.seed(seed)
                    torch.manual_seed(seed)
                    depth_pil = self.perspective_transform(depth_pil)
                else:
                    # Аффинная трансформация
                    seed = random.randint(0, 2**31)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    rgb_pil = self.affine_transform(rgb_pil)
                    
                    random.seed(seed)
                    torch.manual_seed(seed)
                    depth_pil = self.affine_transform(depth_pil)
                
                # Преобразуем обратно в тензоры
                rgb = transforms.ToTensor()(rgb_pil)
                rgb = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(rgb)
                
                depth = transforms.ToTensor()(depth_pil)
            except Exception as e:
                print(f"Ошибка при перспективной аугментации: {e}")
        
        # Добавляем небольшой шум для лучшей устойчивости к квантизации
        if self.noise_augmentation and random.random() > 0.7:
            try:
                # Шум для RGB (малая амплитуда)
                rgb_noise = torch.randn_like(rgb) * 0.01
                rgb = torch.clamp(rgb + rgb_noise, -2.0, 2.0)  # Ограничиваем диапазон с учетом нормализации
                
                # Шум для depth (калиброванной амплитуды)
                depth_noise = torch.randn_like(depth) * 0.005
                depth = torch.clamp(depth + depth_noise, 0.001, 0.999)  # Гарантируем валидные значения глубины
            except Exception as e:
                print(f"Ошибка при шумовой аугментации: {e}")
        
        return rgb, depth
    
    def __getitem__(self, idx):
        """Получение элемента датасета с оптимизированным использованием памяти"""
        # Сначала проверяем кеш
        if self.enable_cache and idx in self.cache:
            self.cache_hits += 1
            
            # Перемещаем элемент в конец OrderedDict (недавно использованный)
            rgb_tensor, depth_tensor = self.cache.pop(idx)
            self.cache[idx] = (rgb_tensor, depth_tensor)
            
            # Применяем аугментацию если нужно
            if self.enable_augmentation:
                rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': self.rgb_files[idx],
                'depth_path': self.depth_files[idx],
                'from_cache': True
            }
        
        # Кеш-промах - загружаем с диска
        self.cache_misses += 1
        
        try:
            # Загружаем RGB изображение
            rgb_path = self.rgb_files[idx]
            rgb_tensor = self.rgb_transform_base(self._load_and_resize_image(rgb_path))
            
            # Загружаем depth изображение
            depth_path = self.depth_files[idx]
            depth_tensor = self.depth_transform_base(self._load_and_resize_image(depth_path, is_depth=True))
            
            # Нормализуем depth с защитой от нулевых значений для лучшей квантизации
            if self.normalize_depth:
                depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
                if depth_max > depth_min:  # Проверяем на вырожденный случай
                    depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min + 1e-8)
                depth_tensor = torch.clamp(depth_tensor, 0.001, 0.999)  # Предотвращаем экстремальные значения
            
            # Гарантируем одноканальный depth
            if depth_tensor.shape[0] > 1:
                depth_tensor = depth_tensor[0].unsqueeze(0)
            
            # Применяем аугментацию если нужно
            if self.enable_augmentation:
                rgb_tensor, depth_tensor = self._apply_augmentation(rgb_tensor, depth_tensor)
            
            # Добавляем в кеш
            if self.enable_cache:
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
            print(f"Ошибка при загрузке или обработке изображения {idx}: {e}")
            
            # Создаем пустые тензоры при ошибке
            rgb_tensor = torch.zeros(3, self.image_size, self.image_size)
            depth_tensor = torch.zeros(1, self.image_size, self.image_size)
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'rgb_path': self.rgb_files[idx] if idx < len(self.rgb_files) else "invalid_path",
                'depth_path': self.depth_files[idx] if idx < len(self.depth_files) else "invalid_path",
                'from_cache': False,
                'error': True
            }
    
    def _load_and_resize_image(self, path, is_depth=False):
        """Оптимизированная загрузка и ресайз изображения с обработкой ошибок"""
        try:
            img = Image.open(path)
            if not is_depth:
                img = img.convert('RGB')
            return img.resize((self.image_size, self.image_size), 
                            resample=Image.NEAREST if is_depth else Image.BILINEAR)
        except Exception as e:
            print(f"Ошибка загрузки изображения {path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            mode = 'L' if is_depth else 'RGB'
            return Image.new(mode, (self.image_size, self.image_size))
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getstate__(self):
        """Кастомный getstate для сериализации"""
        state = self.__dict__.copy()
        # Удаляем несериализуемые объекты
        state['cache'] = OrderedDict()
        state['cache_hits'] = 0
        state['cache_misses'] = 0
        return state
    
    def __setstate__(self, state):
        """Кастомный setstate для десериализации"""
        self.__dict__.update(state)
        # Переинициализируем кеш
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Переинициализируем аугментации если нужно
        if self.enable_augmentation:
            self._setup_enhanced_augmentations()

def create_improved_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=0, 
                           persistent_workers=False, drop_last=False, worker_init_fn=None,
                           prefetch_factor=2):
    """
    Создает оптимизированный DataLoader с настройками для лучшей 
    производительности и устойчивости при обучении моделей для квантизации.
    
    Аргументы:
        dataset: Датасет
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        pin_memory: Использовать ли закрепленную память для более быстрой передачи на GPU
        num_workers: Количество процессов-рабочих (0 = только основной поток)
        persistent_workers: Сохранять ли рабочие процессы между эпохами
        drop_last: Отбрасывать ли последний неполный батч
        worker_init_fn: Функция инициализации воркера
        prefetch_factor: Множитель предварительной загрузки данных
    """
    # Оптимизируем batch_size для GTX 1080
    if dataset[0]['rgb'].shape[1] > 320 or dataset[0]['rgb'].shape[2] > 320:
        batch_size = min(batch_size, 4)  # Уменьшаем размер батча для больших изображений
    elif batch_size > 8 and torch.cuda.is_available():
        # Проверяем доступную память GPU
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_mem_gb = free_mem / (1024**3)
        
        if free_mem_gb < 7.0:  # Для GTX 1080 с 8GB
            batch_size = 8     # Ограничиваем для сохранения памяти на операции
    
    # Используем модульную функцию seed_worker если не указана другая
    if worker_init_fn is None:
        worker_init_fn = seed_worker
    
    # Оптимизируем количество рабочих для максимальной производительности
    if num_workers > 0:
        available_cores = psutil.cpu_count(logical=False)
        if num_workers > available_cores:
            # Не используем больше физических ядер
            num_workers = available_cores
    
    # Создаем загрузчик данных с оптимизированными параметрами
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),  # Закрепляем память только при наличии CUDA
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return loader

class QATPrefetcher:
    """
    Оптимизированный Prefetcher без двойного GPU-буфера:
    - Загружает следующий батч в CPU-RAM;
    - При .next() синхронно копирует CPU→GPU только этот батч;
    - На GPU одновременно находится только один батч.
    
    Идеален для работы с ограниченной памятью GPU (например, GTX 1080 8GB).
    """
    def __init__(self, loader, device="cuda"):
        self._base_loader = loader            # сохраняем для сброса
        self.device = device
        self._loader_iter = iter(loader)      # собственный итератор
        self._next_batch_cpu = None           # для хранения в RAM
        self._preload()                       # загружаем первый батч (CPU)
    
    def _preload(self):
        """Загружает следующий батч в CPU-память (без копий на GPU)"""
        try:
            self._next_batch_cpu = next(self._loader_iter)
        except StopIteration:
            self._next_batch_cpu = None       # эпоха закончилась
        except Exception as e:
            print(f"Ошибка при загрузке батча: {e}")
            self._next_batch_cpu = None
    
    def next(self):
        """
        Возвращает словарь с данными.
        
        Данные остаются на CPU; перенос на GPU выполняется в train-loop:
        rgb = batch['rgb'].to(device, non_blocking=True)
        
        Это обеспечивает хранение на GPU только одного батча одновременно.
        
        Returns:
            dict | None: Батч данных на CPU или None, если эпоха закончилась.
        """
        # Если данных больше нет - сигнал об окончании эпохи
        if self._next_batch_cpu is None:
            return None
        
        # Забираем подготовленный батч и сразу предзагружаем следующий
        batch_cpu = self._next_batch_cpu
        self._preload()
        
        # Никакого копирования на GPU здесь! 
        # Батч вернется в train-loop, где его и перенесут на device
        return batch_cpu
    
    def reset(self):
        """Запустить эпоху заново (создает новый итератор DataLoader)"""
        self._loader_iter = iter(self._base_loader)
        self._next_batch_cpu = None
        self._preload()
