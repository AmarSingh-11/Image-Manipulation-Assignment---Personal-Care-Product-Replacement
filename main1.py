# Image Manipulation Assignment - Personal Care Product Replacement

## Step 1: Set Up Environment
# Install required packages (run this in terminal)
# sudo apt update && sudo apt install python3-pip
# pip3 install opencv-python pillow torch torchvision diffusers transformers rembg

# Step 2: Generate Image Mask
import cv2
import numpy as np
from rembg import remove
from PIL import Image

# Load the image
image_path = "background_image.webp"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use edge detection to identify contours
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour, assuming it's the product in the woman's hand
contours = sorted(contours, key=cv2.contourArea, reverse=True)
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])

    # Extract and remove the product from the image using inpainting
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite("inpainted_image.png", inpainted_image)
    print("Inpainted image created successfully.")

    product_bbox = (x, y, w, h)
else:
    product_bbox = None
    print("No product detected.")

# Step 3: Generate New Product Image
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
prompt = "A realistic personal care cream box, high quality, designed for placement in a woman's hand."
new_product_image = pipe(prompt).images[0]
new_product_image.save("new_product.png")
print("New product image generated successfully.")

# Step 4: Replace Product in Image
# Load inpainted background and new product image
background = cv2.imread("inpainted_image.png")
new_product = cv2.imread("new_product.png")

# Resize the new product to match the original product's dimensions
if product_bbox:
    x, y, w, h = product_bbox
    new_product_resized = cv2.resize(new_product, (w, h))

    # Overlay the new product at the exact location of the old product
    background[y:y+h, x:x+w] = new_product_resized

    # Save the final blended image
    cv2.imwrite("final_replaced_product.png", background)
    print("Product replaced successfully in 'final_replaced_product.png'.")
else:
    print("No product area detected. Unable to replace.")

# Step 5: Refine the Image
blurred = cv2.GaussianBlur(background, (5, 5), 0)
cv2.imwrite("refined_image.png", blurred)
print("Final refined image saved as 'refined_image.png'.")

# Step 6: Display Final Image
cv2.imshow("Final Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

