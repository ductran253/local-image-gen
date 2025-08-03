# Dynamic package installation
import subprocess
import sys
import importlib.util

def install_package(package_names, index_url=None):
    """Install package(s) using pip"""
    # Handle both single package and multiple packages
    if isinstance(package_names, str):
        packages = package_names.split()
    else:
        packages = package_names
    
    print(f"Installing {' '.join(packages)}...")
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if index_url:
        cmd.extend(["--index-url", index_url])
   
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {' '.join(packages)} installed successfully")
        return True
    else:
        print(f"✗ Failed to install {' '.join(packages)}: {result.stderr}")
        return False

def check_cuda_available():
    """Check if CUDA is available on the system"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def ensure_ml_packages():
    """Ensure required ML packages are installed"""
    # Check if torch is already installed
    if importlib.util.find_spec("torch") is None:
        # Determine PyTorch version based on CUDA availability
        if check_cuda_available():
            print("CUDA detected - installing PyTorch with CUDA support")
            torch_packages = ["torch", "torchvision", "torchaudio"]
            torch_index = "https://download.pytorch.org/whl/cu118"
        else:
            print("No CUDA detected - installing CPU-only PyTorch")
            torch_packages = ["torch", "torchvision", "torchaudio"]
            torch_index = "https://download.pytorch.org/whl/cpu"
       
        if not install_package(torch_packages, torch_index):
            raise Exception("Failed to install PyTorch")
    else:
        print("✓ PyTorch is available")
   
    # Check other packages
    other_packages = [
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("tokenizers", "tokenizers"),
    ]
   
    missing_packages = []
    for check_name, install_name in other_packages:
        if importlib.util.find_spec(check_name) is None:
            missing_packages.append(install_name)
   
    if missing_packages:
        print("Installing additional ML packages...")
        for package in missing_packages:
            if not install_package([package]):  # Pass as list for consistency
                raise Exception(f"Failed to install {package}")
        print("✓ All ML packages installed successfully!")
    else:
        print("✓ All ML packages are available")

# Rest of your code remains the same
import os
import sys

if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "model")
os.makedirs(model_path, exist_ok=True)
pipeline = None

def is_pipeline_loaded():
    return pipeline is not None

def load_pipeline():
    from diffusers import StableDiffusionXLPipeline
    import torch
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    global pipeline
    if pipeline is None:
        print(f"Loading model to: {model_path}")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=model_path,
            low_cpu_mem_usage=True,
            device_map="balanced"
        )
        pipeline.enable_vae_tiling()
        pipeline.enable_attention_slicing()
    return pipeline

def generate_image(prompt, output_path="output.png"):
    import torch
    import gc
   
    pipe = load_pipeline()
    result = pipe(prompt, num_inference_steps=30)
    image = result.images[0]
    image.save(output_path)
    
    del result
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_path