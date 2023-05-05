from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


class model:
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-2-1-base"

        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def text2img(self, prompt="a photo of an astronaut riding a horse on mars"):
        image = self.pipe(prompt).images[0]

        image.save(f"outputs/{prompt}.png")
