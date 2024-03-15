# import required libraries and modules
import cv2
import numpy as np

def reduce_resolution(image, block_size):
    # Get dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the number of blocks along rows and columns
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    
    # Create a new image to store the result
    result = np.zeros_like(image)
    
    # Iterate through each block
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            roi = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            # Calculate the average value of the block
            average_value = np.mean(roi)
            
            # Fill the corresponding region in the result image with the average value
            result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = average_value
    
    return result

# Load the image
image = cv2.imread('./images/x-ray.jpg', cv2.IMREAD_GRAYSCALE)

# Reduce resolution for 3x3 blocks
reduced_image_3 = reduce_resolution(image, 3)

# Reduce resolution for 5x5 blocks
reduced_image_5 = reduce_resolution(image, 5)

# Reduce resolution for 7x7 blocks
reduced_image_7 = reduce_resolution(image, 7)

# saving the resulting images
cv2.imwrite('reduced_image_3.jpg', reduced_image_3)
cv2.imwrite('reduced_image_5.jpg', reduced_image_5)
cv2.imwrite('reduced_image_7.jpg', reduced_image_7)

# Display the original and reduced images
cv2.imshow('Original Image', image)
cv2.imshow('Reduced Image (3x3)', reduced_image_3)
cv2.imshow('Reduced Image (5x5)', reduced_image_5)
cv2.imshow('Reduced Image (7x7)', reduced_image_7)
cv2.waitKey(0)
cv2.destroyAllWindows()
