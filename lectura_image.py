import cv2 # Librería para procesamiento de imágenes
import os # Librería para manejar archivos y directorios

ruta_dataset = "dataset"
total_imagenes = 0

for clase in os.listdir(ruta_dataset):
# Construir la ruta completa a la carpeta de la clase
    ruta_clase = os.path.join(ruta_dataset, clase)
# Verificar si es un directorio (carpeta de clase)
    if not os.path.isdir(ruta_clase):
        continue
# Inicializar el contador de imágenes para la clase actual
    contador = 0
# Recorrer cada imagen en la carpeta de la clase
    for archivo in os.listdir(ruta_clase):
# Construir la ruta completa a la imagen
        ruta_imagen = os.path.join(ruta_clase, archivo)
#  Cargar la imagen utilizando OpenCV
        imagen = cv2.imread(ruta_imagen)
# Verificar si la imagen se cargó correctamente
        if imagen is None:
            print("Error al cargar:", archivo)
            continue
#   Redimensionar la imagen a un tamaño fijo (opcional)
        contador += 1 # Incrementar el contador de imágenes para la clase actual
        total_imagenes += 1 # Incrementar el contador total de imágenes

    print(clase, ":", contador, "imágenes")

print("Total de imágenes:", total_imagenes)

