import torch
import sys
import os
import platform
import subprocess
import shutil

def run_command(command):
    """Run a command and return its output, cross-platform compatible."""
    try:
        # Use shell=True for Windows compatibility with some commands
        is_windows = platform.system() == "Windows"
        result = subprocess.run(command, shell=is_windows, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Command failed with code {result.returncode}"
    except Exception as e:
        return f"Error running command: {e}"

def find_executable(name):
    """Find an executable in PATH, cross-platform compatible."""
    executable = shutil.which(name)
    return executable

def check_cuda():
    print("=== PyTorch CUDA Diagnostic ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available through PyTorch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # CUDA version as reported by PyTorch
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        
        # Number of GPUs
        print(f"GPU count: {torch.cuda.device_count()}")
        
        # GPU name
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("PyTorch cannot find CUDA. Let's check the system:")
    
    # Environment variables
    print("\n=== CUDA Environment Variables ===")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME/CUDA_PATH: {cuda_home}")
    
    # Check PATH for CUDA
    path = os.environ.get('PATH', '')
    path_sep = ";" if platform.system() == "Windows" else ":"
    cuda_in_path = False
    
    if path:
        path_entries = path.split(path_sep)
        for entry in path_entries:
            if 'cuda' in entry.lower():
                print(f"CUDA entry in PATH: {entry}")
                cuda_in_path = True
    
    if not cuda_in_path:
        print("No CUDA directories found in PATH")
    
    # Check LD_LIBRARY_PATH on Linux, or PATH on Windows for CUDA libraries
    if platform.system() == "Windows":
        print("\nChecking PATH for CUDA libraries:")
        lib_paths = path_entries
        lib_var_name = "PATH"
    else:
        print("\nLD_LIBRARY_PATH:")
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        lib_paths = ld_library_path.split(':') if ld_library_path else []
        lib_var_name = "LD_LIBRARY_PATH"
    
    cuda_in_lib_path = False
    for entry in lib_paths:
        if 'cuda' in entry.lower():
            print(f"CUDA entry in {lib_var_name}: {entry}")
            cuda_in_lib_path = True
    
    if not cuda_in_lib_path:
        print(f"No CUDA directories found in {lib_var_name}")
    
    # Check if nvidia-smi is available
    print("\n=== NVIDIA System Management Interface ===")
    nvidia_smi_path = find_executable("nvidia-smi")
    if nvidia_smi_path:
        print(f"nvidia-smi found at: {nvidia_smi_path}")
        # Run nvidia-smi
        smi_output = run_command(["nvidia-smi"])
        print("nvidia-smi output:")
        print(smi_output)
    else:
        print("nvidia-smi not found in PATH")
    
    # Check for CUDA toolkit
    print("\n=== CUDA Toolkit ===")
    nvcc_path = find_executable("nvcc")
    if nvcc_path:
        print(f"nvcc found at: {nvcc_path}")
        nvcc_version = run_command(["nvcc", "--version"])
        print("nvcc version:")
        print(nvcc_version)
    else:
        print("nvcc not found in PATH")
    
    # Check for libcudart in a platform-specific way
    print("\n=== CUDA Runtime Library ===")
    if platform.system() == "Windows":
        # On Windows, check a few common locations
        potential_paths = []
        if cuda_home:
            potential_paths.append(os.path.join(cuda_home, "bin", "cudart64_*.dll"))
            potential_paths.append(os.path.join(cuda_home, "lib", "x64", "cudart*.dll"))
        
        for potential_path in potential_paths:
            # Use glob pattern to find files
            import glob
            matches = glob.glob(potential_path)
            if matches:
                print(f"CUDA Runtime found at: {matches}")
                break
        else:
            print("Could not locate CUDA Runtime Library in common locations")
    else:
        # On Linux, use ldconfig if available
        ldconfig_path = find_executable("ldconfig")
        if ldconfig_path:
            libcudart = run_command(["ldconfig", "-p"]) 
            if "libcudart" in libcudart:
                print("CUDA Runtime Library found:")
                for line in libcudart.splitlines():
                    if "libcudart" in line:
                        print(line)
            else:
                print("CUDA Runtime Library not found with ldconfig")
        else:
            # Try locate command as a backup
            locate_path = find_executable("locate")
            if locate_path:
                locate_result = run_command(["locate", "libcudart"])
                if locate_result and "Error" not in locate_result:
                    print("CUDA Runtime Library possibly found at:")
                    print(locate_result)
                else:
                    print("CUDA Runtime Library not found with locate")
            else:
                print("Neither ldconfig nor locate available to search for libcudart")
    
    # Check PyTorch installation details
    print("\n=== PyTorch Installation Details ===")
    try:
        torch_file = torch.__file__
        print(f"PyTorch is installed at: {torch_file}")
        
        # Check PyTorch installation type
        if "+cu" in torch.__version__:
            cuda_version = torch.__version__.split("+cu")[1]
            print(f"PyTorch built with CUDA {cuda_version}")
        elif "+cpu" in torch.__version__:
            print("PyTorch CPU-only build detected")
        else:
            # For newer PyTorch that doesn't expose CUDA version in __version__
            cuda_enabled = True if torch.version.cuda else False
            print(f"PyTorch CUDA support: {'Yes' if cuda_enabled else 'No'}")
    except Exception as e:
        print(f"Error checking PyTorch installation: {e}")
    
    print("\n=== Recommendations ===")
    if not torch.cuda.is_available():
        print("1. Check if your PyTorch installation includes CUDA support:")
        if platform.system() == "Windows":
            print("   You may need to reinstall PyTorch with CUDA:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("   (Use cu121 for CUDA 12.x compatibility)")
        else:
            print("   You may need to reinstall PyTorch with CUDA:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n2. Make sure your NVIDIA drivers are properly installed")
        print("\n3. Check if your GPU supports CUDA")
        
        print("\n4. Ensure environment variables are set correctly:")
        if platform.system() == "Windows":
            print("   - CUDA_PATH should point to your CUDA installation (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x)")
            print("   - PATH should include CUDA bin directory")
        else:
            print("   - CUDA_HOME should point to your CUDA installation")
            print("   - PATH should include CUDA bin directory")
            print("   - LD_LIBRARY_PATH should include CUDA lib64 directory")
        
        if platform.system() == "Linux":
            print("\n5. If running in Docker, make sure the container has GPU access:")
            print("   - Use the --gpus flag: docker run --gpus all ...")
            print("   - Make sure nvidia-docker is installed if needed")

if __name__ == "__main__":
    check_cuda()
