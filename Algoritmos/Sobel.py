from google.colab.patches import cv2_imshow
import cv2
import numpy as np

image_path = '/content/lena.png'  
image = cv2.imread(image_path)


if image is None:
    print(f"Error: Could not load image from {image_path}. Please check the file path and ensure the image exists.")
    exit()

imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #transformando em escala de cinza
cv2_imshow(imagegray)

sobelX = cv2.Sobel(imagegray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(imagegray, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2_imshow(sobelX)
cv2_imshow(sobelY)
cv2_imshow(sobelCombined)
cv2.waitKey(0)
