/*
 * Depth Model Inference for TH1520 NPU using STB for image processing
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <debug.h>
#include <model.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <dirent.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>
#include "npu_model/io.h"
#include "shl_ref.h"
#include "shl_c920.h"
//#include "npu_model/shl_ref.h"
 
 #define MIN(x, y) ((x) < (y) ? (x) : (y))
 #define MAX(x, y) ((x) > (y) ? (x) : (y))
 #define MAX_PATH_LEN 1024
 #define MAX_FILES 500
 
 // Configuration for your model
 #define RESIZE_HEIGHT 256
 #define RESIZE_WIDTH 256
 
 // Function prototypes
 void *create_graph(const char *params_path);
 float *preprocess_image(const char *image_path, int *width, int *height);
 int postprocess_depth(float *depth_data, int depth_width, int depth_height,
                      const char *out_path, int orig_width, int orig_height);
 int is_image_file(const char *filename);
 void create_directory(const char *path);
 uint64_t get_time_us();
 
 // Helper function to get time in microseconds
 uint64_t get_time_us() {
     struct timeval tv;
     gettimeofday(&tv, NULL);
     return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
 }
 
 // Check if a file is an image based on extension
 int is_image_file(const char *filename) {
     const char *ext = strrchr(filename, '.');
     if (!ext) return 0;
     
     ext++;  // Skip the dot
     return (strcasecmp(ext, "jpg") == 0 || 
             strcasecmp(ext, "jpeg") == 0 || 
             strcasecmp(ext, "png") == 0);
 }
 
 // Create directory if it doesn't exist
 void create_directory(const char *path) {
     struct stat st = {0};
     if (stat(path, &st) == -1) {
         if (mkdir(path, 0755) == -1) {
             printf("Failed to create directory %s: %s\n", path, strerror(errno));
         } else {
             printf("Created output directory: %s\n", path);
         }
     }
 }

 // Функция для печати массива байт в hex (отладка)
 static void print_hex(const unsigned char *data, size_t length)
 {
     for (size_t i = 0; i < length; i++) {
         printf("%02X ", data[i]);
         if ((i + 1) % 16 == 0)
             printf("\n");
     }
     if (length % 16 != 0)
         printf("\n");
 }

 /*
  * create_graph:
  *   - Загружает модель (bm или params),
  *   - Вызывает csinn_() или csinn_import_binary_model() (внутри model.c),
  *   - Возвращает указатель на сессию.
  */
 void *create_graph(const char *model_file)
 {
     DEBUG_PRINT("Открытие файла модели: %s\n", model_file);
     int binary_size = 0;
     char *params = get_binary_from_file(model_file, &binary_size);
     if (!params) {
         DEBUG_PRINT("[ERROR] Не удалось загрузить файл модели.\n");
         return NULL;
     }
     
     DEBUG_PRINT("Файл модели загружен, размер: %d байт\n", binary_size);
     DEBUG_PRINT("Первые 128 байт файла:\n");
     print_hex((unsigned char *)params, (binary_size < 128 ? binary_size : 128));
 
     int path_len = strlen(model_file);
     if (path_len >= 7) {
         const char *suffix_params = model_file + (path_len - 7);
         DEBUG_PRINT("Проверка расширения .params: %s\n", suffix_params);
         if (strcmp(suffix_params, ".params") == 0) {
             DEBUG_PRINT("Файл имеет расширение .params, вызываем csinn_()\n");
             void *sess = csinn_(params);
             DEBUG_PRINT("csinn_() вернул sess=%p\n", sess);
             return sess;
         }
     }
     if (path_len >= 3) {
         const char *suffix_bm = model_file + (path_len - 3);
         DEBUG_PRINT("Проверка расширения .bm: %s\n", suffix_bm);
         if (strcmp(suffix_bm, ".bm") == 0) {
             DEBUG_PRINT("Файл имеет расширение .bm, обрабатываем как HHB бинарную модель\n");
             if (binary_size < 4128 + sizeof(struct shl_bm_sections)) {
                 DEBUG_PRINT("[ERROR] Файл слишком маленький для секций.\n");
                 shl_mem_free(params);
                 return NULL;
             }
             struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
             DEBUG_PRINT("Получены секции модели: graph_offset = %d, params_offset = %d\n",
                         section->graph_offset, section->params_offset);
             if (section->graph_offset) {
                 DEBUG_PRINT("graph_offset != 0 -> csinn_import_binary_model()\n");
                 void *sess = csinn_import_binary_model(params);
                 DEBUG_PRINT("csinn_import_binary_model() вернул: %p\n", sess);
                 return sess;
             } else {
                 DEBUG_PRINT("graph_offset равен 0, вызываем csinn_() с смещением: %d * 4096 байт\n", section->params_offset);
                 void *sess = csinn_(params + section->params_offset * 4096);
                 DEBUG_PRINT("csinn_() (с смещением) вернул: %p\n", sess);
                 return sess;
             }
         }
     }
     DEBUG_PRINT("[ERROR] Неподдерживаемый формат файла модели.\n");
     shl_mem_free(params);
     return NULL;
 }

 // Preprocess image using STB
 float *preprocess_image(const char *image_path, int *width, int *height) {
     int channels, orig_width, orig_height;
     DEBUG_PRINT("--- Start preprocessing image: %s\n", image_path);
     // Load image with STB
     unsigned char *img_data = stbi_load(image_path, &orig_width, &orig_height, &channels, 3);
     if (!img_data) {
         printf("Failed to load image: %s\n", image_path);
         return NULL;
     }
     
     printf("Loaded image: %s (%dx%d, %d channels)\n", image_path, orig_width, orig_height, channels);
     
     // Save original dimensions for output
     if (width) *width = orig_width;
     if (height) *height = orig_height;
     
     // Resize image to model input dimensions
     unsigned char *resized = (unsigned char*)malloc(RESIZE_WIDTH * RESIZE_HEIGHT * 3);
     if (!resized) {
         printf("Failed to allocate memory for resized image\n");
         stbi_image_free(img_data);
         return NULL;
     }
     
     // Resize with STB
     stbir_resize_uint8_linear(img_data, orig_width, orig_height, 0,
                       resized, RESIZE_WIDTH, RESIZE_HEIGHT, 0, 3);
     
     // Free original image data
     stbi_image_free(img_data);
     
     // Convert to float and normalize (divide by 255)
     float *float_data = (float*)malloc(RESIZE_WIDTH * RESIZE_HEIGHT * 3 * sizeof(float));
     if (!float_data) {
         printf("Failed to allocate memory for float data\n");
         free(resized);
         return NULL;
     }
     
    //  // Normalize and convert to NCHW format (needed for NPU)
    //  for (int c = 0; c < 3; c++) {
    //      for (int h = 0; h < RESIZE_HEIGHT; h++) {
    //          for (int w = 0; w < RESIZE_WIDTH; w++) {
    //              float_data[c * RESIZE_HEIGHT * RESIZE_WIDTH + h * RESIZE_WIDTH + w] = 
    //                  resized[(h * RESIZE_WIDTH + w) * 3 + c] / 255.0f;
    //          }
    //      }
    //  }

    // Нормализация и конвертация в NCHW формат с правильной нормализацией ImageNet
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < RESIZE_HEIGHT; h++) {
            for (int w = 0; w < RESIZE_WIDTH; w++) {
                float pixel_value = resized[(h * RESIZE_WIDTH + w) * 3 + c] / 255.0f;
                // Применяем нормализацию ImageNet
                float_data[c * RESIZE_HEIGHT * RESIZE_WIDTH + h * RESIZE_WIDTH + w] = 
                    (pixel_value - mean[c]) / std[c];
            }
        }
    }

     DEBUG_PRINT("Finish preprocessed image data: %p\n", float_data);
     free(resized);
     return float_data;
 }
 
void dump_depth_histogram(float *depth_data, int size, const char *filename) {
    // Простейшая гистограмма для анализа распределения значений
    int bins[10] = {0};
    
    for (int i = 0; i < size; i++) {
        int bin = (int)(depth_data[i] * 10);
        if (bin >= 0 && bin < 10) bins[bin]++;
    }
    
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "Depth value histogram:\n");
        for (int i = 0; i < 10; i++) {
            fprintf(f, "%.1f-%.1f: %d\n", i*0.1f, (i+1)*0.1f, bins[i]);
        }
        fclose(f);
    }
}
void save_raw_depth_data(float *depth_data, int width, int height, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (f) {
        fwrite(depth_data, sizeof(float), width * height, f);
        fclose(f);
        printf("Raw depth data saved to %s\n", filename);
    } else {
        printf("Failed to save raw depth data\n");
    }
}

void apply_median_filter(float* depth_data, int width, int height, float* filtered) {
    // Простой медианный фильтр 3×3 для удаления шума
    const int window_size = 3;
    const int half_window = window_size / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float window[window_size * window_size];
            int count = 0;
            
            for (int dy = -half_window; dy <= half_window; dy++) {
                for (int dx = -half_window; dx <= half_window; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        window[count++] = depth_data[ny * width + nx];
                    }
                }
            }
            
            // Простая сортировка для нахождения медианы
            for (int i = 0; i < count - 1; i++) {
                for (int j = i + 1; j < count; j++) {
                    if (window[i] > window[j]) {
                        float temp = window[i];
                        window[i] = window[j];
                        window[j] = temp;
                    }
                }
            }
            
            filtered[y * width + x] = window[count / 2];
        }
    }
}

 // Post-process depth map and save as image
 int postprocess_depth(float *depth_data, int depth_width, int depth_height,
                      const char *out_path, int orig_width, int orig_height) {
    DEBUG_PRINT("Start postprocessing depth data: %p\n", depth_data);
    float* filtered_depth = malloc(depth_width * depth_height * sizeof(float));
    apply_median_filter(depth_data, depth_width, depth_height, filtered_depth);


     // Allocate memory for depth visualization
     unsigned char *depth_vis = (unsigned char*)malloc(depth_width * depth_height * 3);
     if (!depth_vis) {
         printf("Failed to allocate memory for depth visualization\n");
         return -1;
     }
     
     // Find min and max for normalization
     float min_val = 1.0f, max_val = 0.0f;
     for (int i = 0; i < depth_width * depth_height; i++) {
         min_val = MIN(min_val, filtered_depth[i]);
         max_val = MAX(max_val, filtered_depth[i]);
     }
    // Выводим диапазон значений для отладки
    DEBUG_PRINT("Depth values range: min=%.6f, max=%.6f\n", min_val, max_val);

    //  // Normalize and colorize depth map (using a simple blue-to-red gradient)
    //  float range = max_val - min_val;
    //  if (range < 0.001f) range = 1.0f;  // Avoid division by zero
     
    //  for (int i = 0; i < depth_width * depth_height; i++) {
    //      float normalized = (depth_data[i] - min_val) / range;
    //      normalized = normalized < 0.0f ? 0.0f : (normalized > 1.0f ? 1.0f : normalized);
         
    //      // Simple blue (far) to red (near) colormap
    //      depth_vis[i * 3 + 0] = (unsigned char)(normalized * 255.0f);  // R
    //      depth_vis[i * 3 + 1] = 0;  // G
    //      depth_vis[i * 3 + 2] = (unsigned char)((1.0f - normalized) * 255.0f);  // B
    //  }
    
    // MobileDepth использует sigmoid, поэтому значения уже в диапазоне [0,1]
    // и дополнительная нормализация не требуется
    for (int i = 0; i < depth_width * depth_height; i++) {
        float value = filtered_depth[i];
        
        // Можно инвертировать, если нужно (1.0 - value)
        // В зависимости от желаемой интерпретации: 
        // 0 = далеко, 1 = близко или наоборот
        
        // Colormap: синий (далеко) -> красный (близко)
        // Если значение уже в [0,1], дополнительная нормализация не нужна
        unsigned char r = (unsigned char)(value * 255.0f);
        unsigned char b = (unsigned char)((1.0f - value) * 255.0f);
        
        depth_vis[i * 3 + 0] = r;  // R
        depth_vis[i * 3 + 1] = 0;  // G
        depth_vis[i * 3 + 2] = b;  // B
    }

    // If original dimensions are different, resize back
    unsigned char *final_vis = depth_vis;
    // Проверяем, нужно ли изменять размер изображения    
    if (orig_width != depth_width || orig_height != depth_height) {
        // Используем корректные размеры для изменения размера изображения
        // в соответствии с ожидаемыми входными данными модели
        
        // Глубина должна быть изменена до размеров оригинального изображения
        final_vis = (unsigned char*)malloc(orig_width * orig_height * 3);
        if (!final_vis) {
            printf("Failed to allocate memory for resized output\n");
            free(depth_vis);
            return -1;
        }
        
        stbir_resize_uint8_linear(depth_vis, depth_width, depth_height, 0,
                          final_vis, orig_width, orig_height, 0, 3);
    }

    // Save as PNG
     int result = stbi_write_png(out_path, 
                                (final_vis == depth_vis) ? depth_width : orig_width,
                                (final_vis == depth_vis) ? depth_height : orig_height,
                                3, final_vis, 0);
     
     if (result == 0) {
         printf("Failed to write output image: %s\n", out_path);
     } else {
         printf("Depth map saved to: %s\n", out_path);
     }

     dump_depth_histogram(filtered_depth, depth_width * depth_height, "depth_histogram.txt");
     save_raw_depth_data(filtered_depth, depth_width, depth_height, "raw_depth.bin");
     // Clean up
     if (final_vis != depth_vis) {
         free(final_vis);
     }
     free(depth_vis);
     DEBUG_PRINT("Finish postprocessed depth data: %p\n", filtered_depth);
     return result ? 0 : -1;
 }
 
// Функция process_image с DMA-совместимым выделением памяти
int process_image(const char *model_path, const char *input_path, const char *output_path) {
    int orig_width, orig_height;
    uint64_t start_time, preprocess_time, inference_time, postprocess_time;

    // Load model
    start_time = get_time_us();
    void *sess = create_graph(model_path);
    if (!sess) {
        printf("Failed to load model from %s\n", model_path);
        return -1;
    }
    printf("Model loaded in %.3f ms\n", (get_time_us() - start_time) / 1000.0);

    // Preprocess image
    start_time = get_time_us();
    float *input_data_f32 = preprocess_image(input_path, &orig_width, &orig_height);
    if (!input_data_f32) {
        printf("Failed to preprocess image\n");
        csinn_session_deinit(sess);
        csinn_free_session(sess);
        return -1;
    }
    preprocess_time = get_time_us() - start_time;

    // Convert input dtype
    int8_t *input_data_temp = shl_ref_f32_to_input_dtype(0, input_data_f32, sess);
    free(input_data_f32);

    // Setup input tensor
    struct csinn_tensor *input_tensor = csinn_alloc_tensor(NULL);
    input_tensor->mtype = CSINN_MEM_TYPE_DMABUF;
    input_tensor->dim_count = 4;
    input_tensor->dim[0] = 1;
    input_tensor->dim[1] = 3;
    input_tensor->dim[2] = RESIZE_HEIGHT;
    input_tensor->dim[3] = RESIZE_WIDTH;
    input_tensor->dtype = CSINN_DTYPE_INT8;

    // DMA-совместимое выделение памяти
    int input_size_bytes = csinn_tensor_byte_size(input_tensor);        // размер данных тензора в байтах, с учётом dtype
    input_tensor->data = shl_mem_alloc_aligned(input_size_bytes, 0);
    if (!input_tensor->data) {
        printf("Failed to allocate DMA-compatible memory\n");
        csinn_free_tensor(input_tensor);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
        return -1;
    }
    memcpy(input_tensor->data, input_data_temp, input_size_bytes);
    shl_mem_free(input_data_temp);

    // Run inference
    inference_time = get_time_us();
    struct csinn_tensor *input_tensors[1] = {input_tensor};
    csinn_update_input_and_run(input_tensors, sess);
    inference_time = get_time_us() - inference_time;

    // Get output tensor
    start_time = get_time_us();
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->data = NULL;
    csinn_get_output(0, output, sess);
    // Выводим информацию о выходном тензоре для отладки
    DEBUG_PRINT("Output tensor: dtype=%d, layout=%d, dim=[%d,%d,%d,%d]\n", 
        output->dtype, output->layout, 
        output->dim[0], output->dim[1], output->dim[2], output->dim[3]);
    DEBUG_PRINT("Output tensor: min=%f, max=%f, scale=%f, zero_point=%d\n",
        output->qinfo->min, output->qinfo->max, 
        output->qinfo->scale, output->qinfo->zero_point);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    // Проверяем результат преобразования
    if (!foutput || !foutput->data) {
        printf("Error: Failed to transform output tensor to float32\n");
        // Обработка ошибки...
        return -1;
    }

    float *depth_data = (float*)foutput->data;
    int num_pixels = output->dim[2] * output->dim[3];
    // Проверяем диапазон значений
    float min_val = 1.0f, max_val = 0.0f;
    for (int i = 0; i < num_pixels; i++) {
        min_val = MIN(min_val, depth_data[i]);
        max_val = MAX(max_val, depth_data[i]);
    }
    DEBUG_PRINT("Raw depth values: min=%.6f, max=%.6f\n", min_val, max_val);

    int result = postprocess_depth(foutput->data,
                                   output->dim[3],
                                   output->dim[2],
                                   output_path,
                                   orig_width,
                                   orig_height);

    postprocess_time = get_time_us() - start_time;

    printf("Performance stats for %s:\n", input_path);
    printf("  Preprocessing: %.3f ms\n", preprocess_time / 1000.0);
    printf("  Inference:     %.3f ms\n", inference_time / 1000.0);
    printf("  Postprocessing: %.3f ms\n", postprocess_time / 1000.0);
    printf("  Total:         %.3f ms\n", (preprocess_time + inference_time + postprocess_time) / 1000.0);

    shl_ref_tensor_transform_free_f32(foutput);
    csinn_free_tensor(output);
    shl_mem_free(input_tensor->data);
    csinn_free_tensor(input_tensor);
    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return result;
}

 
// Функция process_directory с DMA-совместимым выделением памяти
int process_directory(const char *model_path, const char *input_dir, const char *output_dir) {
    DIR *dir;
    struct dirent *entry;
    char input_files[MAX_FILES][MAX_PATH_LEN];
    char input_path[MAX_PATH_LEN];
    char output_path[MAX_PATH_LEN];
    int file_count = 0;

    create_directory(output_dir);

    if ((dir = opendir(input_dir)) == NULL) {
        printf("Cannot open directory %s: %s\n", input_dir, strerror(errno));
        return -1;
    }

    while ((entry = readdir(dir)) != NULL && file_count < MAX_FILES) {
        if (is_image_file(entry->d_name)) {
            snprintf(input_files[file_count], MAX_PATH_LEN, "%s", entry->d_name);
            file_count++;
        }
    }
    closedir(dir);

    if (file_count == 0) {
        printf("No image files found in %s\n", input_dir);
        return -1;
    }

    printf("Found %d image files to process\n", file_count);

    void *sess = create_graph(model_path);
    if (!sess) {
        printf("Failed to load model from %s\n", model_path);
        return -1;
    }

    uint64_t total_start_time = get_time_us();
    uint64_t total_inference_time = 0;

    int success_count = 0;
    for (int i = 0; i < file_count; i++) {
        snprintf(input_path, MAX_PATH_LEN, "%s/%s", input_dir, input_files[i]);

        char base_name[MAX_PATH_LEN];
        strcpy(base_name, input_files[i]);
        char *dot = strrchr(base_name, '.');
        if (dot) *dot = '\0';

        snprintf(output_path, MAX_PATH_LEN, "%s/%s_depth.png", output_dir, base_name);

        printf("\nProcessing [%d/%d]: %s -> %s\n", i+1, file_count, input_path, output_path);

        int orig_width, orig_height;
        uint64_t start_time = get_time_us();
        float *input_data_f32 = preprocess_image(input_path, &orig_width, &orig_height);
        if (!input_data_f32) {
            printf("Failed to preprocess image, skipping\n");
            continue;
        }
        uint64_t preprocess_time = get_time_us() - start_time;

        int8_t *input_data_temp = shl_ref_f32_to_input_dtype(0, input_data_f32, sess);
        free(input_data_f32);

        // ==========================
        // Setup input tensor
        struct csinn_tensor *input_tensor = csinn_alloc_tensor(NULL);
        input_tensor->mtype = CSINN_MEM_TYPE_DMABUF;
        input_tensor->dim_count = 4;
        input_tensor->dim[0] = 1;
        input_tensor->dim[1] = 3;
        input_tensor->dim[2] = RESIZE_HEIGHT;
        input_tensor->dim[3] = RESIZE_WIDTH;
        input_tensor->dtype = CSINN_DTYPE_INT8;

        // DMA-совместимое выделение памяти
        int input_size_bytes = csinn_tensor_byte_size(input_tensor);        // размер данных тензора в байтах, с учётом dtype
        input_tensor->data = shl_mem_alloc_aligned(input_size_bytes, 0);
        if (!input_tensor->data) {
            printf("Failed to allocate DMA-compatible memory\n");
            csinn_free_tensor(input_tensor);
            csinn_session_deinit(sess);
            csinn_free_session(sess);
            return -1;
        }
        memcpy(input_tensor->data, input_data_temp, input_size_bytes);
        shl_mem_free(input_data_temp);
        // ==========================

        // Run inference
        uint64_t inference_start = get_time_us();
        csinn_update_input_and_run(&input_tensor, sess);
        uint64_t inference_time = get_time_us() - inference_start;
        total_inference_time += inference_time;

        uint64_t postproc_start = get_time_us();
        struct csinn_tensor *output = csinn_alloc_tensor(NULL);
        output->data = NULL;
        csinn_get_output(0, output, sess);

        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);

        int result = postprocess_depth(foutput->data,
                                     output->dim[3],
                                     output->dim[2],
                                     output_path,
                                     orig_width,
                                     orig_height);
        
        uint64_t postprocess_time = get_time_us() - postproc_start;

        printf("  Preprocessing: %.3f ms\n", preprocess_time / 1000.0);
        printf("  Inference:     %.3f ms\n", inference_time / 1000.0);
        printf("  Postprocessing: %.3f ms\n", postprocess_time / 1000.0);
        printf("  Total:         %.3f ms\n", 
               (preprocess_time + inference_time + postprocess_time) / 1000.0);

        shl_ref_tensor_transform_free_f32(foutput);
        csinn_free_tensor(output);
        shl_mem_free(input_tensor->data);
        csinn_free_tensor(input_tensor);

        if (result == 0) success_count++;
    }

    uint64_t total_time = get_time_us() - total_start_time;

    printf("\n===== Processing Summary =====\n");
    printf("Successfully processed %d/%d images\n", success_count, file_count);
    printf("Average inference time: %.3f ms\n", (total_inference_time / file_count) / 1000.0);
    printf("Total processing time: %.3f ms (%.3f seconds)\n", 
       total_time / 1000.0, total_time / 1000000.0);
    printf("Output depth maps saved to: %s\n", output_dir);
    printf("=============================\n");

    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return success_count;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  %s <model_file.bm> <input_image> <output_image>\n", argv[0]);
        printf("  %s <model_file.bm> <input_directory> <output_directory>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *input_path = argv[2];
    const char *output_path = argv[3];

    struct stat path_stat;
    stat(input_path, &path_stat);

    if (S_ISDIR(path_stat.st_mode)) {
        // Input is a directory, process all images
        return process_directory(model_path, input_path, output_path);
    } else {
        // Input is a single file
        return process_image(model_path, input_path, output_path);
    }
}
