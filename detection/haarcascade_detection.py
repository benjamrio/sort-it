import cv2
from detection.traitement_rectangles import fusion_rectangle
# En entrée le path de l'image qu'on souhaite détecter
# En sortie on affiche un rectangle rouge autour de l'image, basé sur la cascade xml crée, données recoltées avec l'api


#Lire la documentation de live_haarcascade_detection pour comprendre précisemment ce programme
def haarcascade_detection(filename,scaleFactor,minNeighbours):
    cascade = cv2.CascadeClassifier('Data/cascadeGarbage.xml')
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray,scaleFactor,minNeighbours)

    # Amélioration de la détection par fusion
    (x,y,w,h) = fusion_rectangle(objects)

    # Tracé des rectangles et extraction des objets
    extract = img[y:y+h, x:x+w]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imwrite('Data/newPicture.jpg', extract)


if __name__ == "__main__":
    haarcascade_detection("C:/Users/Benjamin/Desktop/glassBottle.jpg",1.2,5)