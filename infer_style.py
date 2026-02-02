# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import time
from ip_adapter import my_IPAdapterXL as IPAdapterXL

base_model_path = "/home/lyl/stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/home/lyl/IP-Adapter-main/sdxl_models/image_encoder"
ip_ckpt = "/home/lyl/IP-Adapter-main/sdxl_models/ip-adapter_sdxl.bin"
device = "cuda:0"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

image = "./style/9.jpg"
image = Image.open(image)
image.resize((512, 512))
start = time.time()
# generate image
images = ip_model.generate(pil_image=image,
                           prompt = "a blue cup",
                           negative_prompt= "",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=50,
                           seed=42,
                          )
end = time.time()
print("推理时间:", end - start, "秒")
print(images[0])
images[0].save("result.png")