import cv2

# Load images
background = cv2.imread("background_image.png")
product = cv2.imread("new_product.png")

# Resize the product to fit
product_resized = cv2.resize(product, (100, 100))

# Position the product (adjust coordinates as needed)
background[100:200, 150:250] = product_resized

# Save the final image
cv2.imwrite("final_image.png", background)

print("Product replaced successfully in 'final_image.png'.")

