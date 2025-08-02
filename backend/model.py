from diffusers import StableDiffusionXLPipeline
import torch
import os
import sys

torch.backends.cuda.matmul.allow_tf32 = True

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(os.path.join(base_path, "../model"))

# Initialize pipeline as None
pipeline = None

def is_pipeline_loaded():
    """Check if pipeline is loaded"""
    return pipeline is not None

def load_pipeline():
    """Load the pipeline if not already loaded"""
    global pipeline
    if pipeline is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
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
    
    # Load pipeline if not loaded
    pipe = load_pipeline()
    
    # Run inference
    result = pipe(prompt)
    image = result.images[0]
    image.save(output_path)

    # Cleanup
    del result
    torch.cuda.empty_cache()
    gc.collect()

    return output_path