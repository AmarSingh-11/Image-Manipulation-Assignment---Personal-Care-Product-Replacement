from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

prompt = "A personal care cream box, high quality, realistic"
image = pipe(prompt).images[0]

image.save("new_product.png")

print("New product image generated as 'new_product.png'.")

