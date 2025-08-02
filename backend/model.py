from diffusers import AutoPipelineForText2Image
import torch
import os
import sys

torch.backends.cuda.matmul.allow_tf32 = True

if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(os.path.join(base_path, "../model"))

max_memory = {0:"8GB", "cpu":"8GB"}
#Load Model

#Lazy Loading approach
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=model_path,
    low_cpu_mem_usage=False
).to("cuda")

pipeline.enable_model_cpu_offload()
pipeline.enable_vae_tiling()
pipeline.enable_attention_slicing()

def generate_image(prompt, output_path="output.png"):
    image = pipeline(prompt, num_inference_steps=30).images[0]
    image.save(output_path)
    return output_path