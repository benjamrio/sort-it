import cv2
import numpy as np
import haarcascade_detection 
from traitement_rectangles import fusion_rectangle
import pytest
##########################################################################################
#Nous testons ici la qualité du fichier XML en introduisant une notion pertinent de coût #
##########################################################################################

#Une première fonction qui calcule la distance euclidienne entre deux points
def dst_2_pts(a,b):
    (x1,y1) = a
    (x2,y2) = b
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

'''
Nous allons créer une foncton qui évalue la qualité de la detection haarcascade 
Idée : on introduit la définition d'un coût entre deux positions de rectangles.
Le cout sera la somme des distances entre les coins correspondants
'''
#En entrée, le path de deux fichiers : une image avec un dechet dessus, et la même image
# avec le déchet entouré (par l'humain) dessus et pos_rect =(x,y,w,h) les infos du rectangles "manuel"
def coutRaisonnable(filename,pos_rect_th):

    cascade = cv2.CascadeClassifier('Data/cascadeGarbage.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, 1.2, 5)

    # Amélioration de la détection par fusion et obtention du rectangle final
    (x_obt,y_obt,w_obt,h_obt) = fusion_rectangle(objects)
    
    #Obtention des infos du rectangle théorique
    (x_th,y_th,w_th,h_th) = pos_rect_th
    
    #Calcul du cout (somme des distances entre les coins)
    tl_obt, dl_obt, br_obt, tr_obt = (x_obt,y_obt), (x_obt,y_obt+h_obt), (x_obt+w_obt,y_obt+h_obt), (x_obt+w_obt,y_obt)
    tl_th, dl_th, br_th, tr_th = (x_th,y_th), (x_th,y_th+h_th), (x_th+w_th,y_th+h_th), (x_th+w_th,y_th)

    cout = dst_2_pts(tl_obt,tl_th) + dst_2_pts(dl_obt,dl_th) + dst_2_pts(br_obt,br_th) + dst_2_pts(tr_obt,tr_th)

    assert est_Raisonnable(img.shape, cout)
    

#on définit une notion de raisonnable :
# est raisonnable un cout qui est inférieur à la demi-somme de largeur et la largeur de l'image

def est_Raisonnable(shape,cout):
    height, width, channels = shape
    return(cout<(height + width)/2)


if __name__=="__main__":
    #a l'aide d'un éditeur d'images, on détermine les coordonnées des rectangles tracés à la main
    coutRaisonnable("Data/glass140.jpg",(119,108,200,152))