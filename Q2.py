# import required modules and libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('./images/noise_image.jpg', cv2.IMREAD_GRAYSCALE)

def average_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to load the image.")
    
else:
    # Perform 3x3 spatial average
    average3x3_image = average_filter(image, 3)

    # Perform 10x10 spatial average
    average10x10_image = average_filter(image, 10)

    # Perform 20x20 spatial average
    average20x20_image = average_filter(image, 20)

    # Display the original and filtered images
    plt.subplot(221),plt.imshow(image, cmap='gray'),plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(222),plt.imshow(average3x3_image, cmap='gray'),plt.title('3x3 Average Filtered Image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(223),plt.imshow(average10x10_image, cmap='gray'),plt.title('10x10 Average Filtered Image')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(224),plt.imshow(average20x20_image, cmap='gray'),plt.title('20x20 Average Filtered Image')
    plt.xticks([]), plt.yticks([])
    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    