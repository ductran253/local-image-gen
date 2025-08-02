from diffusers import AutoPipelineForText2Image
import torch
import os

torch.backends.cuda.matmul.allow_tf32 = True

# Path to ../model relative to current script (main/backend/)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))

max_memory = {0:"16GB", 1:"16GB"}
#Load Model
pipeline = None  # global variable to hold the model

def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=model_path,
            low_cpu_mem_usage=False,
            max_memory=max_memory
        ).to("cuda")
        #Optimization
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_tiling()
        pipeline.enable_attention_slicing()


def generate_image(prompt: str):
    if pipeline is None:
        load_pipeline()
    image = pipeline(prompt, num_inference_steps=30).images[0]
    return image
