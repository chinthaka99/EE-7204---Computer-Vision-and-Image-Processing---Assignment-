# import required modules and libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# function to reduce the intensity level
def reduce_intensity_levels(image, num_levels):
  # Check if num_levels is a power of 2
  if not (num_levels & (num_levels - 1) == 0):
    raise ValueError("num_levels must be a power of 2.")

  # Convert image to float for normalization
  image = image.astype(np.float32)

  # Calculate the normalization factor based on the number of levels
  normalization_factor = 255 / (num_levels - 1)

  # Quantize the image by rounding and scaling
  reduced_image = np.round(image / normalization_factor) * normalization_factor

  # Convert back to uint8 for image display
  reduced_image = reduced_image.astype(np.uint8)

  return reduced_image

# read the image
image = cv2.imread('./images/lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Choose the desired number of intensity levels
num_levels = 2

reduced_image = reduce_intensity_levels(image, num_levels)

# Display the original and reduced images
plt.subplot(121),plt.imshow(image, cmap='gray'),plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(reduced_image, cmap='gray'),plt.title('Intensity Levels Reduced Image')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
