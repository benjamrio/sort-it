import numpy as np
import cv2

#En entrée un path
#Retourne l'image encodée selon cv2 ie un array

def load(filename):
    img  = cv2.imread(filename)
    return img

#En entrée une image selon cv2
#En sortie affiche l'image dans une windows que l'on ferme manuellement
def display_image(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#En entrée un filename(path)
#En sortie affiche l'image, se ferme quand on la ferme
def load_and_display_image(filename):
    img = load(filename)
    display_image(img)
