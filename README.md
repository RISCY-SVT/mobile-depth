# MobileDepth: Efficient Depth Estimation for Mobile and Edge Devices

A high-performance implementation of a lightweight depth estimation model designed for mobile and edge devices. This repository contains a complete pipeline for training and deploying the MobileDepth model on the MidAir synthetic dataset.

![MobileDepth Example](https://github.com/RISCY-SVT/mobile-depth/assets/example_output.jpg)

## Features

- **Lightweight Architecture**: MobileDepth uses depthwise separable convolutions to achieve excellent depth estimation with minimal parameters (3.2M)
- **Memory Optimizations**: RAM-based caching system for ultra-fast data loading
- **Multi-scale Output**: Multiple prediction scales with deep supervision for improved gradient flow
- **High-Performance Training**: Optimized for systems with large RAM and multi-core CPUs
- **Mixed Precision Support**: Accelerated training with FP16 operations
- **Attention Mechanisms**: Spatial attention modules improve focus on relevant areas
- **System Tests**: Comprehensive system benchmark tools to optimize for your hardware
- **Dataset Viewer**: Interactive tool for exploring and visualizing the MidAir dataset

## Model Architecture

MobileDepth uses an encoder-decoder architecture:

- **Encoder**: Based on MobileNet-v1 with depthwise separable convolutions
- **Skip Connections**: Feature maps from different levels are connected with attention
- **Decoder**: Improved upsampling blocks with residual connections
- **Multi-scale Output**: Three output scales during training for deep supervision
- **Activation**: ReLU6 for efficient quantization support

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- 8+ GB GPU RAM recommended
- 32+ GB system RAM recommended for full dataset caching

### Dependencies

```
torch>=2.0
torchvision>=0.15
numpy
matplotlib
pillow
tqdm
psutil
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RISCY-SVT/mobile-depth.git
cd mobile-depth
```

2. Install dependencies:
```bash
python -v venv depth

# For Windows commandline
.\depth\Scripts\activate.bat

# For Linux 
source ./depth/bin/activate

pip install -r requirements.txt
```

3. Download the [MidAir dataset](https://midair.ulg.ac.be/download.html) and extract it to your preferred location.

## Dataset Setup

The MidAir dataset should be structured as follows:

```
MidAir_dataset/
├── Kite_training/
│   ├── sunny/
│   │   ├── color_left/
│   │   │   ├── trajectory_XXXX/
│   │   │   │   ├── *.JPEG files
│   │   ├── depth/
│   │   │   ├── trajectory_XXXX/
│   │   │   │   ├── *.png files
│   ├── foggy/
│   ├── cloudy/
│   ├── sunset/
├── Kite_test/
├── PLE_training/
├── PLE_test/
└── VO_test/
```

## Usage

### Basic Training

```bash
python optimized_main.py --data_root MidAir_dataset --datasets Kite_training --environments sunny --batch_size 12 --image_size 384 --mixed_precision
```

### Advanced Training with Optimizations

```bash
python optimized_main.py --data_root MidAir_dataset --datasets Kite_training --environments sunny foggy cloudy sunset --batch_size 12 --image_size 384 --num_workers 16 --prefetch_factor 4 --cache_size 50000 --mixed_precision --output_dir MidAir_trained
```

### Resume Training from Checkpoint

```bash
python optimized_main.py --data_root MidAir_dataset --datasets Kite_training --environments sunny --resume output/checkpoints/model_epoch_5.pth
```

### Testing the Model

```bash
python test_depth_model.py --model_path output/checkpoints/best_model.pth --image_size 640
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_root` | Path to MidAir dataset | `MidAir_dataset` |
| `--datasets` | Dataset types to use | `['Kite_training']` |
| `--environments` | Environments to include | `['sunny']` |
| `--batch_size` | Batch size for training | `12` |
| `--epochs` | Number of training epochs | `10` |
| `--lr` | Learning rate | `0.001` |
| `--image_size` | Training image resolution | `384` |
| `--val_freq` | Validation frequency (epochs) | `1` |
| `--save_freq` | Checkpoint saving frequency | `1` |
| `--output_dir` | Output directory | `output` |
| `--num_workers` | Number of data loading workers | `16` |
| `--mixed_precision` | Use mixed precision training | `False` |
| `--resume` | Path to checkpoint for resuming | `None` |
| `--cache_size` | Maximum images in RAM cache | `38374` |
| `--prefetch_factor` | Number of batches to prefetch | `4` |

## Performance Optimizations

The repository includes several optimizations for training:

1. **RAM Caching**: Stores processed images in RAM to avoid repeated disk I/O
2. **Multi-threaded Prefetching**: Loads and processes data in background threads 
3. **Mixed Precision Training**: Uses FP16 calculations where possible for 2-3x speed increase
4. **Attention Mechanisms**: Spatial attention helps the model focus on relevant features
5. **Automatic Memory Management**: Strategic GPU memory cleanup prevents fragmentation

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce `batch_size`
- Decrease `image_size`
- Reduce `cache_size`

### Training Too Slow

To improve training speed:
- Enable `--mixed_precision`
- Increase `--num_workers` (for multi-core systems)
- Ensure your dataset is on a fast storage device

### CUDA Misaligned Address

This error typically occurs during validation:
- Ensure you're using the latest optimized validation function
- Try reducing validation batch size
- Add additional `torch.cuda.empty_cache()` calls

## Results

The model achieves the following performance on the MidAir test set:

| Metric | Value |
|--------|-------|
| RMSE | 0.074 |
| AbsRel | 0.62 |
| δ < 1.25 | 0.88 |
| Model Size | 12.9 MB |
| Inference Time (GTX 1080) | 11.8 ms |

## Citation

If you use this code in your research, please cite our work:

```
@misc{mobileDepth2025,
  author = {RISCY-SVT},
  title = {MobileDepth: Efficient Depth Estimation for Mobile and Edge Devices},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/RISCY-SVT/mobile-depth}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MidAir dataset from [University of Liège](https://midair.ulg.ac.be/)
- Inspired by MobileNet architecture by Google
- Base depth estimation principles from [MonoDepth](https://github.com/mrharicot/monodepth)