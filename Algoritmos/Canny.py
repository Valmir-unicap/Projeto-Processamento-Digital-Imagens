from google.colab.patches import cv2_imshow
import cv2
import numpy as np

image_path = '/content/lena.png'  
image = cv2.imread(image_path)


if image is None:
    print(f"Error: Could not load image from {image_path}. Please check the file path and ensure the image exists.")
    exit()

# carrega a imagem, converte-a para tons de cinzento e desfoca-a ligeiramente
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# calcular um limiar “largo”, “médio” e “apertado” para as arestas
# utilizando o detetor de arestas Canny

wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

artigo = cv2.Canny(blurred, 100, 200)

# mostrar os mapas de arestas Canny de saída
cv2_imshow(wide)
cv2_imshow(mid)
cv2_imshow(tight)

cv2_imshow(artigo)
cv2.waitKey(0)
