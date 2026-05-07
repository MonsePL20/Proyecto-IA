import cv2
import os

ruta_dataset = "dataset"
total_imagenes = 0

for clase in os.listdir(ruta_dataset):

    ruta_clase = os.path.join(ruta_dataset, clase)

    if not os.path.isdir(ruta_clase):
        continue

    contador = 0

    for archivo in os.listdir(ruta_clase):

        ruta_imagen = os.path.join(ruta_clase, archivo)

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            print("Error al cargar:", archivo)
            continue

        contador += 1
        total_imagenes += 1

    print(clase, ":", contador, "imágenes")

print("Total de imágenes:", total_imagenes)