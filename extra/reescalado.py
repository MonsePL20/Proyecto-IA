# Pre-procesado de imágenes para el dataset de gatos y perros todo el dataset
# Este script realiza varias transformaciones en las imágenes del dataset, incluyendo:
# - Reescalado a un tamaño uniforme (160x160 píxeles)


import cv2
import os

# Ruta de tu dataset
ruta_dataset = "dataset"

total_imagenes = 0

print("Leyendo dataset...\n")

# Recorrer cada carpeta (clase)
for clase in os.listdir(ruta_dataset):
    ruta_clase = os.path.join(ruta_dataset, clase)

    if not os.path.isdir(ruta_clase):
        continue

    contador_clase = 0
    print(f"Clase: {clase}")

    # Recorrer imágenes dentro de la clase
    for archivo in os.listdir(ruta_clase):
        ruta_imagen = os.path.join(ruta_clase, archivo)

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            print(f" Error al cargar: {ruta_imagen}")
            continue

        # Redimensionar
        imagen_rescalada = cv2.resize(imagen, (160, 160))

        # VISUALIZACIÓN EN TIEMPO REAL

        # Combinar original + rescalada
        original_peq = cv2.resize(imagen, (160, 160))
        combinado = cv2.hconcat([original_peq, imagen_rescalada])

        cv2.imshow("Escaneo en proceso (Original vs Rescalada)", combinado)

        # Espera corta (50 ms)
        if cv2.waitKey(50) & 0xFF == 27:  # tecla ESC para salir
            print("Proceso detenido por el usuario")
            break

        contador_clase += 1
        total_imagenes += 1

    print(f"Imágenes leídas: {contador_clase}\n")

print(f"TOTAL DE IMÁGENES: {total_imagenes}")

# Cerrar ventanas
cv2.destroyAllWindows()