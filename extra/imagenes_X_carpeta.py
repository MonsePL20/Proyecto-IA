# Ejemplo de lectura y visualización de imágenes desde una carpeta usando OpenCV
# solo muestrando la imagen sin texto ni contadores

import cv2
import os

# Ruta del dataset (elige cualquiera de las carpetas de tu dataset)
ruta = "dataset/Bombay_cat"

# Recorrer cada imagen en la carpeta
for archivo in os.listdir(ruta):
    img = cv2.imread(os.path.join(ruta, archivo))
    img = cv2.resize(img, (128,128))


# Mostrar la imagen
    cv2.imshow("Imagen", img)
    cv2.waitKey(60)  # Espera 60 ms entre imágenes

cv2.destroyAllWindows()


