import cv2
import os

ruta = "dataset/Calico_cat"

for archivo in os.listdir(ruta):

    ruta_imagen = os.path.join(ruta, archivo)

    # Leer imagen
    img = cv2.imread(ruta_imagen)

    if img is None:
        print("Error al cargar:", archivo)
        continue

    # Imagen original
    original = cv2.resize(ruta_imagen)

    # Imagen escalada
    escalada = cv2.resize(img, (120,120))

    # Mostrar ventanas separadas
    cv2.imshow("Imagen Original", original)
    cv2.imshow("Imagen Escalada", escalada)

    # Posicionar ventanas una al lado de otra
    cv2.moveWindow("Imagen Original", 100, 100)
    cv2.moveWindow("Imagen Escalada", 350, 100)

    cv2.waitKey(60)

cv2.destroyAllWindows()

