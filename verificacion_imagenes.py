import cv2
import os

ruta = "dataset/Calico_cat"

for archivo in os.listdir(ruta):
    img = cv2.imread(os.path.join(ruta, archivo))
    img = cv2.resize(img, (160,160))
    
    cv2.imshow("Imagen", img)
    cv2.waitKey(60)

cv2.destroyAllWindows()


