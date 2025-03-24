import os
import time
import sys
import argparse
import random
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import psutil
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path

def print_system_info():
    """Print detailed system information"""
    print("\n========== SYSTEM INFORMATION ==========")
    print(f"CPU: {multiprocessing.cpu_count()} cores")
    
    # Memory
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
    
    # Disk where data is located
    data_path = os.path.abspath("MidAir_dataset")
    if os.path.exists(data_path):
        try:
            disk_usage = psutil.disk_usage(data_path)
            print(f"Disk ({data_path}): {disk_usage.total / (1024**3):.2f} GB total, "
                 f"{disk_usage.free / (1024**3):.2f} GB free")
        except:
            print(f"Disk info not available for {data_path}")
    
    # PyTorch and CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Test CUDA performance
        print("\nRunning quick CUDA performance test...")
        test_cuda_performance()
    
    # Python information
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")

def test_cuda_performance():
    """Run a quick CUDA performance benchmark"""
    # Create a reasonably sized tensor
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Warm up
    for _ in range(5):
        torch.matmul(x, y)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    iters = 20
    for _ in range(iters):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Matrix multiplication time: {elapsed/iters*1000:.2f} ms per iteration")
    
    # CNN benchmark
    model = models.resnet18().cuda()
    model.eval()
    input_tensor = torch.randn(16, 3, 224, 224, device='cuda')
    
    # Warm up
    for _ in range(3):
        model(input_tensor)
        
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    iters = 10
    for _ in range(iters):
        model(input_tensor)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"ResNet-18 inference time: {elapsed/iters*1000:.2f} ms per batch")

def test_disk_performance(data_dir="MidAir_dataset"):
    """Benchmark disk performance"""
    print("\n========== DISK PERFORMANCE TEST ==========")
    
    # Find some image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                image_files.append(os.path.join(root, file))
                if len(image_files) >= 100:
                    break
        if len(image_files) >= 100:
            break
    
    if not image_files:
        print("No image files found for disk performance test")
        return
    
    # Sequential read test
    print(f"Testing sequential read speed (reading {len(image_files)} images)...")
    start = time.time()
    total_size = 0
    
    for img_path in image_files:
        with open(img_path, 'rb') as f:
            data = f.read()
            total_size += len(data)
    
    elapsed = time.time() - start
    read_speed = total_size / elapsed / (1024 * 1024)
    print(f"Sequential read speed: {read_speed:.2f} MB/s")
    
    # Random read test
    print("Testing random read performance...")
    random.shuffle(image_files)
    start = time.time()
    total_size = 0
    
    for img_path in image_files:
        with open(img_path, 'rb') as f:
            data = f.read()
            total_size += len(data)
    
    elapsed = time.time() - start
    read_speed = total_size / elapsed / (1024 * 1024)
    print(f"Random read speed: {read_speed:.2f} MB/s")
    
    # Image loading test
    print("Testing image loading and processing...")
    start = time.time()
    
    for i, img_path in enumerate(image_files[:20]):
        img = Image.open(img_path)
        img = img.resize((320, 320))
        img = np.array(img)
    
    elapsed = time.time() - start
    print(f"Average image load+resize time: {elapsed/min(20, len(image_files))*1000:.2f} ms per image")
    
    return read_speed

# This should be at the top level of the script, not inside any function
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img
    
def test_pytorch_dataloader(data_dir="MidAir_dataset", num_workers_options=[1, 4, 8, 16]):
    """Test different DataLoader configurations"""
    print("\n========== DATALOADER PERFORMANCE TEST ==========")
    
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    # Find image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg')):
                image_files.append(os.path.join(root, file))
                if len(image_files) >= 200:
                    break
        if len(image_files) >= 200:
            break
    
    if not image_files:
        print("No image files found for dataloader test")
        return
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = SimpleImageDataset(image_files, transform=transform)
    
    # Test different num_workers settings
    results = {}
    batch_size = 16
    
    for num_workers in num_workers_options:
        print(f"Testing DataLoader with num_workers={num_workers}...")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        start = time.time()
        batch_times = []
        batch_start = time.time()  # Initialize before the loop
        
        for i, batch in enumerate(dataloader):
            # Calculate time for current batch (except first batch)
            if i > 0:
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
            # Break after enough batches    
            if i >= 5:
                break
                
            # Set start time for next batch
            batch_start = time.time() 
               
        
        elapsed = time.time() - start
        
        # Calculate statistics
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            results[num_workers] = avg_batch_time
            print(f"  Average batch loading time: {avg_batch_time*1000:.2f} ms")
        else:
            print("  Not enough batches to measure time")
    
    # Plot results
    if results:
        plt.figure(figsize=(10, 5))
        workers = list(results.keys())
        times = [results[w]*1000 for w in workers]  # Convert to ms
        
        plt.bar(workers, times)
        plt.xlabel('Number of Workers')
        plt.ylabel('Batch Loading Time (ms)')
        plt.title('DataLoader Performance by Workers')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        os.makedirs("benchmark_results", exist_ok=True)
        plt.savefig("benchmark_results/dataloader_benchmark.png")
        plt.close()
        
        # Find optimal number of workers
        optimal_workers = min(results, key=results.get)
        print(f"\nOptimal number of workers for your system: {optimal_workers}")
        return optimal_workers
    
    return 4  # Default

def optimize_system():
    """Apply system optimizations based on benchmark results"""
    print("\n========== SYSTEM OPTIMIZATION ==========")
    
    # Create optimization results directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Generate optimization recommendations
    recommendations = {
        "batch_size": 8,
        "num_workers": 4,
        "image_size": 320,
        "cache_size": 10000,
        "mixed_precision": True
    }
    
    # Adjust based on GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory > 10:
            recommendations["batch_size"] = 16
            recommendations["image_size"] = 512
        elif gpu_memory > 6:
            recommendations["batch_size"] = 12
            recommendations["image_size"] = 384
    
    # Adjust based on system RAM
    system_ram = psutil.virtual_memory().total / (1024**3)
    
    if system_ram > 100:
        recommendations["cache_size"] = min(100000, int(system_ram * 300))
    elif system_ram > 60:
        recommendations["cache_size"] = min(40000, int(system_ram * 300))
    elif system_ram > 30:
        recommendations["cache_size"] = min(20000, int(system_ram * 300))
    
    # Adjust num_workers based on benchmark
    optimal_workers = test_pytorch_dataloader()
    if optimal_workers:
        recommendations["num_workers"] = optimal_workers
    else:
        recommendations["num_workers"] = min(16, multiprocessing.cpu_count())
    
    # Run disk performance test
    disk_speed = test_disk_performance()
    
    # Adjust prefetch factor based on disk speed
    if disk_speed and disk_speed < 50:  # Slow disk
        recommendations["prefetch_factor"] = 6
        print("\nSlow disk detected! Increasing prefetch factor to compensate.")
    else:
        recommendations["prefetch_factor"] = 4
    
    # Print and save recommendations
    print("\n========== OPTIMIZATION RECOMMENDATIONS ==========")
    print("Based on system benchmarks, here are the recommended settings:")
    
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Create a script with optimized command
    cmd = (f"python optimized_main.py "
          f"--data_root MidAir_dataset "
          f"--datasets Kite_training "
          f"--environments sunny "
          f"--batch_size {recommendations['batch_size']} "
          f"--image_size {recommendations['image_size']} "
          f"--num_workers {recommendations['num_workers']} "
          f"--prefetch_factor {recommendations['prefetch_factor']} "
          f"--cache_size {recommendations['cache_size']} "
          f"{'--mixed_precision' if recommendations['mixed_precision'] else ''} "
          f"--output_dir MidAir_optimized")
    
    with open("run_optimized.bat", "w") as f:
        f.write("@echo off\n")
        f.write("echo Running with optimized settings...\n")
        f.write(f"{cmd}\n")
        f.write("pause\n")
    
    print("\nOptimization complete!")
    print("A script with the optimized settings has been created: run_optimized.bat")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='System benchmark and optimization for MobileDepth training')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmarks and only show system info')
    args = parser.parse_args()
    
    print("=== MobileDepth Training System Benchmark ===")
    print_system_info()
    
    if not args.skip_benchmarks:
        optimize_system()
