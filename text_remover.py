import cv2
import numpy as np
import easyocr
from IPython.display import Image
from matplotlib import pyplot as plt
from pylab import rcParams

# Set the figure size for matplotlib
rcParams['figure.figsize'] = 8, 16

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Display the original image
Image("img/01293.png")

# Read the image
image = cv2.imread("img/01293.png")

# Perform text detection using EasyOCR
output = reader.readtext("img/01293.png")

# Create a mask to cover all text regions
mask = np.zeros_like(image[:, :, 0])
for detection in output:
    cord = detection[0]
    x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
    x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
    mask[y_min:y_max, x_min:x_max] = 255

# Inpaint the text regions to remove them from the image
inpaint_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Display the original and inpainted images
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image with Text Removed')
plt.imshow(cv2.cvtColor(inpaint_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
