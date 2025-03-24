import os
import cv2
import time
import argparse
import numpy as np

def get_sorted_file_list(directory, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')):
    exts = tuple(ext.lower() for ext in extensions)
    files = [f for f in os.listdir(directory) if f.lower().endswith(exts)]
    files.sort()
    return files

def main(base_dir):
    """
    Ожидаемая структура датасета:
    base_dir/
      cloudy/
         color_left/   -> файлы JPEG (1024x1024)
         depth/        -> файлы PNG (1024x1024, 16-бит)
      foggy/
         ...
      sunny/
         ...
      sunset/
         ...
    """
    # Получаем список погодных условий
    weather_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    weather_dirs.sort()
    
    # Собираем список всех пар: (weather, trajectory, filename)
    pairs = []
    for weather in weather_dirs:
        weather_path = os.path.join(base_dir, weather)
        color_left_dir = os.path.join(weather_path, "color_left")
        depth_dir = os.path.join(weather_path, "depth")
        if not (os.path.isdir(color_left_dir) and os.path.isdir(depth_dir)):
            continue
        traj_dirs = [d for d in os.listdir(color_left_dir) if os.path.isdir(os.path.join(color_left_dir, d))]
        traj_dirs.sort()
        for traj in traj_dirs:
            traj_color_dir = os.path.join(color_left_dir, traj)
            traj_depth_dir = os.path.join(depth_dir, traj)
            if not os.path.isdir(traj_depth_dir):
                continue
            color_files = get_sorted_file_list(traj_color_dir, extensions=('.jpg', '.jpeg', '.JPEG'))
            for fname in color_files:
                pairs.append((weather, traj, fname))
    
    total_pairs = len(pairs)
    print(f"Найдено пар: {total_pairs}")
    
    mode = "manual"   # Режим: "manual" или "slideshow"
    delay = None      # Задержка в секундах в режиме слайдшоу
    index = 0

    while index < total_pairs:
        weather, traj, fname = pairs[index]
        color_path = os.path.join(base_dir, weather, "color_left", traj, fname)
        depth_path = os.path.join(base_dir, weather, "depth", traj, fname)
        
        # Если файл depth не найден, пробуем заменить расширение на .PNG
        if not os.path.exists(depth_path):
            base_name, ext = os.path.splitext(fname)
            alt_fname = base_name + ".PNG"
            depth_path = os.path.join(base_dir, weather, "depth", traj, alt_fname)
        
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"Пропуск: {color_path} или {depth_path} не существует.")
            index += 1
            continue

        # Загружаем изображения
        color_img = cv2.imread(color_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if color_img is None or depth_img is None:
            print(f"Ошибка загрузки: {color_path} или {depth_path}")
            index += 1
            continue

        # Если depth изображение одноканальное, преобразуем в BGR и применяем цветовую карту
        if len(depth_img.shape) == 2:
            if depth_img.dtype != np.uint8:
                max_val = np.max(depth_img)
                alpha = 1 if max_val == 0 else 255.0 / max_val
                depth_img_norm = cv2.convertScaleAbs(depth_img, alpha=alpha)
            else:
                depth_img_norm = depth_img
            depth_img_vis = cv2.applyColorMap(depth_img_norm, cv2.COLORMAP_RAINBOW)
        else:
            depth_img_vis = depth_img

        # Если размеры не совпадают, меняем размер depth до размеров color
        if color_img.shape[:2] != depth_img_vis.shape[:2]:
            depth_img_vis = cv2.resize(depth_img_vis, (color_img.shape[1], color_img.shape[0]))

        # Объединяем изображения по горизонтали
        combined = cv2.hconcat([color_img, depth_img_vis])
        info_text = f"Weather: {weather}, Trajectory: {traj}, File: {fname}, Pair: {index+1}/{total_pairs}, Mode: {mode}"
        cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Depth Viewer", combined)

        # Получаем нажатую клавишу
        if mode == "manual":
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(delay * 1000))
            if key == -1:
                index += 1
                continue
            key = key & 0xFF

        # Обработка клавиш:
        # Пробел: переключаемся в ручной режим и переходим к следующей паре
        if key == 32:
            mode = "manual"
            index += 1
        # Цифры: задают задержку и переводят в режим слайдшоу
        elif key >= ord('0') and key <= ord('9'):
            digit = key - ord('0')
            if digit == 0:
                delay = 1.0
            elif digit == 1:
                delay = 0.1
            else:
                delay = float(digit/10)
            mode = "slideshow"
            index += 1
        # Клавиша t: перейти к следующей траектории в текущей погодной папке
        elif key == ord('t'):
            current_weather = weather
            current_traj = traj
            # Пропускаем все пары с той же траекторией
            while index < total_pairs and pairs[index][0] == current_weather and pairs[index][1] == current_traj:
                index += 1
        # Клавиша w: перейти к следующей погодной папке
        elif key == ord('w'):
            current_weather = weather
            while index < total_pairs and pairs[index][0] == current_weather:
                index += 1
        # Клавиша q: выход
        elif key == ord('q'):
            print("Выход по нажатию 'q'")
            break
        else:
            mode = "manual"
            index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Интерактивный просмотр пар: цветное изображение (левое) и соответствующая карта глубины")
    parser.add_argument("--data_dir", type=str, required=True, help="Путь к корневой папке датасета")
    args = parser.parse_args()
    main(args.data_dir)
