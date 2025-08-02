from diffusers import AutoPipelineForText2Image
import torch
import os
import sys
from diffusers import StableDiffusionXLPipeline
from diffusers.hooks import apply_group_offloading

torch.backends.cuda.matmul.allow_tf32 = True

if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(os.path.join(base_path, "../model"))

max_memory = {"cuda":"7GB","cpu":"8GB"}
#Load Model

#Lazy Loading approach
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=model_path,
    low_cpu_mem_usage=True,
    max_memory=max_memory
)

pipeline.enable_model_cpu_offload()

pipeline.enable_vae_tiling()

pipeline.enable_attention_slicing()

def generate_image(prompt, output_path="output.png"):
    import torch
    import gc

    # Run inference
    result = pipeline(prompt, num_inference_steps=30)
    image = result.images[0]
    image.save(output_path)

    # Explicitly delete unnecessary stuff
    del result
    torch.cuda.empty_cache()
    gc.collect()

    return output_path