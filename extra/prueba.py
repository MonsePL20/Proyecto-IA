import cv2
import os

# Ruta del dataset (elige cualquiera de las carpetas de tu dataset)
ruta = "dataset/Bombay"

# Recorrer cada imagen en la carpeta
for archivo in os.listdir(ruta):
    img = cv2.imread(os.path.join(ruta, archivo))
    img = cv2.resize(img, (128,128))


# Mostrar la imagen
    cv2.imshow("Imagen", img)
    cv2.waitKey(10)

cv2.destroyAllWindows()