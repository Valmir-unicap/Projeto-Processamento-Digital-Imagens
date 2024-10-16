# Algoritmo - Laplacian

from google.colab.patches import cv2_imshow
import cv2
import numpy as np

image_path = '/content/lena.png'  
image = cv2.imread(image_path)


if image is None:
    print(f"Error: Could not load image from {image_path}. Please check the file path and ensure the image exists.")
else:

    image_blur = cv2.GaussianBlur(image, (3, 3), 0)

    imagegray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    
    cv2_imshow(image)
    cv2_imshow(imagegray)

    lap = cv2.Laplacian(imagegray, cv2.CV_64F)
    cv2_imshow(lap)
    lap2 = np.uint8(np.absolute(lap))
    cv2_imshow(lap2) 
