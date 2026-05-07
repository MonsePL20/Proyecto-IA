# Ejemplo de lectura y visualización de una imagen individual usando OpenCV

import cv2

# Ruta de UNA imagen (elige cualquiera de tu dataset)
ruta_imagen = "dataset/Calico_cat/Calico-1.jpg"

# Leer imagen
imagen = cv2.imread(ruta_imagen)

# Validar
if imagen is None:
    print("Error: no se pudo cargar la imagen")
    exit()

# Redimensionar
imagen_rescalada = cv2.resize(imagen, (160, 160))

# Mostrar imágenes
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Rescalada", imagen_rescalada)

cv2.waitKey(0)
cv2.destroyAllWindows()