import cv2 
import numpy as np 
from scipy import ndimage 
from google.colab.patches import cv2_imshow

image_path = '/content/lena.png'  
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}. Please check the file path and ensure the image exists.")
    exit()

roberts_cross_v = np.array( [[1, 0 ], [0,-1 ]] ) 

roberts_cross_h = np.array( [[ 0, 1 ], [ -1, 0 ]] ) 

# Converter a imagem para escala de cinza antes de aplicar a convolução
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
img = img.astype('float64') 
img/=255.0

vertical = ndimage.convolve( img, roberts_cross_v) 
horizontal = ndimage.convolve( img, roberts_cross_h ) 

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical)) 
edged_img*=255

cv2_imshow(image)
cv2_imshow(edged_img)
