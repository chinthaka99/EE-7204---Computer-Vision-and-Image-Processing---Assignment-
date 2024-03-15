#import required libraries and modules
import cv2
import numpy as np
from matplotlib import pyplot as plt

def rotate_image(image, angle):
  # Get image center
  center = tuple(np.array(image.shape[1::-1]) / 2)

  # Generate rotation matrix
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

  # Define image dimensions after rotation
  new_width, new_height = image.shape[1: : -1]  # Swap width and height for correct calculation
  new_width = int(abs(rot_mat[0, 0] * new_width) + abs(rot_mat[0, 1] * new_height))
  new_height = int(abs(rot_mat[1, 0] * new_width) + abs(rot_mat[1, 1] * new_height))
  
  rot_mat[0, 2] += (new_width - image.shape[1]) / 2
  rot_mat[1, 2] += (new_height - image.shape[0]) / 2
  
  rotated_image = cv2.warpAffine(image, rot_mat, (new_width, new_height))

  return rotated_image

# Load the image
image = cv2.imread('./images/arrow.png')

# Rotate by 45 degrees
rotated_image_45 = rotate_image(image.copy(), 45)

# Rotate by 90 degrees
rotated_image_90 = rotate_image(image.copy(), 90)

plt.subplot(221),plt.imshow(image, cmap='gray'),plt.title('Original Image')
plt.xticks([]), plt.yticks([])
    
plt.subplot(222),plt.imshow(rotated_image_45, cmap='gray'),plt.title('Rotated by 45 Degrees')
plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(rotated_image_90, cmap='gray'),plt.title('Rotated by 90 Degrees')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

