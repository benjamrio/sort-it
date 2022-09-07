import cv2
import matplotlib.pyplot as plt
from detection.traitement_rectangles import fusion_rectangle



def live_detection(scaleFactor,minNeighbours):
    # Chargement du classifier que nous avons construit
    cascade = cv2.CascadeClassifier('Data/cascadeGarbage.xml')

    # Chargement de la vidéo de la webcam 
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        # Lecture d'une image de la vidéo
        _, img = cap.read()
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detection de l'objet avec un facteur d'échelle sigma (conseillé ~ 1,2) 
        # et un nombre de proches de voisins neighbours (conseillé ~ 5).
        objects = cascade.detectMultiScale(gray, scaleFactor,minNeighbours)
        # Amélioration de la détection par fusion

        (x,y,w,h) = fusion_rectangle(objects)
        # Tracé du rectangle et extraction de l'objet
        extract = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Affichage de l'image avec rectangle
        cv2.imshow('img', img)
        
        # Appuyer sur la touche 'a' enregistre les extractions qui seront traitées
        if cv2.waitKey(1) & 0xFF == ord('a'):
            #cv2.imshow('Objets sélectionnés', extract)
            #confirm = input('Confirmer la sélection ? (o/n)')
            #if confirm in ['o', 'oui', 'yes', 'y']:
                #return objects
            if (x,y,w,h) != (0,0,0,0):
                cv2.imwrite('Data/newPicture.jpg', extract)
                #deuxième option : return(img_final) ne fonctionne pas car return stoppe la fonction et on ne lira
                #jamais le cap.release()
                break
            else: 
                print("Pas de rectangle sélectionné") 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_detection(2,6)
