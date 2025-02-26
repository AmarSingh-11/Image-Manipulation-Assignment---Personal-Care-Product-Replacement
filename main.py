# Image Manipulation Assignment - Personal Care Product Replacement

## Step 1: Set Up Environment
# Install required packages (run this in terminal)
# sudo apt update && sudo apt install python3-pip
# pip3 install opencv-python pillow torch torchvision diffusers transformers rembg

# Step 2: Generate Image Mask
import cv2

# Load the image
image = cv2.imread("/home/amar-singh/Desktop/Machine learning/RealTimePredictionWebsite/ai_image_project/background_image.webp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a mask
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Save the mask
cv2.imwrite("product_mask.png", mask)
print("Product mask created successfully.")

# Step 3: Generate New Product Image
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")

# Generate a new product image
prompt = "A realistic personal care cream box"
new_product_image = pipe(prompt).images[0]
new_product_image.save("new_product.png")
print("New product image generated successfully.")

# Step 4: Replace Product in Image
# Load images
product = cv2.imread("new_product.png")

# Resize the product to fit
product_resized = cv2.resize(product, (100, 100))

# Position the product (adjust coordinates as needed)
image[150:250, 200:300] = product_resized

# Save the final image
cv2.imwrite("final_image.png", image)
print("Product replaced successfully in 'final_image.png'.")

# Step 5: Refine the Image
# Apply Gaussian blur for smooth edges
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imwrite("refined_image.png", blurred)
print("Image refined successfully as 'refined_image.png'.")

# Step 6: View Final Image
# Open the final image
cv2.imshow("Refined Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

