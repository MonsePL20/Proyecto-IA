
# Pre-procesado de imágenes para el dataset de gatos y perros

import cv2
import os
import numpy as np

# Ruta principal del dataset
ruta_dataset = "dataset"

# Carpeta donde se guardarán las imágenes procesadas
ruta_salida = "procesadas"

total_imagenes = 0

print("====================================")
print("LEYENDO Y PROCESANDO DATASET...")
print("====================================\n")

# Recorrer carpetas principales
for clase in os.listdir(ruta_dataset):

    ruta_clase = os.path.join(ruta_dataset, clase)

    # Verificar que sea carpeta
    if not os.path.isdir(ruta_clase):
        continue

    print(f"Clase: {clase}")

    contador_clase = 0

    # Crear carpetas de salida
    os.makedirs(os.path.join(ruta_salida, "reescaladas", clase), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, "grises", clase), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, "binarias", clase), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, "bordes", clase), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, "etiquetadas", clase), exist_ok=True)

    # Recorrer imágenes
    for archivo in os.listdir(ruta_clase):

        ruta_imagen = os.path.join(ruta_clase, archivo)

        # Leer imagen
        imagen = cv2.imread(ruta_imagen)

        # Verificar errores
        if imagen is None:
            print(f" Error al cargar: {ruta_imagen}")
            continue

        # ===================================
        # IMAGEN RESCALADA
        # ===================================
        imagen_rescalada = cv2.resize(imagen, (160, 160))

        # ===================================
        # ESCALA DE GRISES
        # ===================================
        gris = cv2.cvtColor(imagen_rescalada, cv2.COLOR_BGR2GRAY)

        # ===================================
        # IMAGEN BINARIA
        # ===================================
        _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

        # ===================================
        # SUAVIZADO + CANNY
        # ===================================
        blur = cv2.GaussianBlur(gris, (5, 5), 0)

        bordes = cv2.Canny(blur, 100, 200)

        # ===================================
        # ETIQUETADO DE OBJETOS
        # ===================================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria)

        etiquetada = imagen_rescalada.copy()

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area >= 500:

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                cv2.rectangle(etiquetada,
                              (x, y),
                              (x + w, y + h),
                              (0, 255, 0),
                              1)

        # ===================================
        # GUARDAR IMÁGENES
        # ===================================

        cv2.imwrite(
            os.path.join(ruta_salida, "reescaladas", clase, archivo),
            imagen_rescalada
        )

        cv2.imwrite(
            os.path.join(ruta_salida, "grises", clase, archivo),
            gris
        )

        cv2.imwrite(
            os.path.join(ruta_salida, "binarias", clase, archivo),
            binaria
        )

        cv2.imwrite(
            os.path.join(ruta_salida, "bordes", clase, archivo),
            bordes
        )

        cv2.imwrite(
            os.path.join(ruta_salida, "etiquetadas", clase, archivo),
            etiquetada
        )

        # ===================================
        # VISUALIZACIÓN EN TIEMPO REAL
        # ===================================

        combinado = cv2.hconcat([
            imagen_rescalada,
            cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR),
            etiquetada
        ])

        cv2.imshow(
            "Original | Gris | Binaria | Bordes | Etiquetada",
            combinado
        )

        # ESC para salir
        if cv2.waitKey(50) & 0xFF == 27:
            print("Proceso detenido por el usuario")
            break

        contador_clase += 1
        total_imagenes += 1

    print(f"Imágenes procesadas: {contador_clase}\n")

print("====================================")
print(f"TOTAL DE IMÁGENES: {total_imagenes}")
print("====================================")

cv2.destroyAllWindows()

