import cv2

# Load the final image
image = cv2.imread("final_image.png")

# Apply a blur to smooth edges
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Save the refined image
cv2.imwrite("refined_image.png", blurred)

print("Image refined and saved as 'refined_image.png'.")

