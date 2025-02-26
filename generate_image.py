from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

# Generate an image
prompt = "A woman applying face cream while holding a cream box"
image = pipe(prompt).images[0]

# Save the image
image.save("background_image.png")

print("Image generated successfully as 'background_image.png'.")

