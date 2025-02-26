import cv2

# Load the image
image = cv2.imread("background_image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a mask
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Save the mask
cv2.imwrite("product_mask.png", mask)

print("Mask created successfully as 'product_mask.png'.")

