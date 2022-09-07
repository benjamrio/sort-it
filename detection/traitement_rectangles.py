import cv2

def ecart_hauteur_largeur(x1,y1,x2,y2):
    h = abs(y2-y1)
    w = abs(x2-x1)
    return(w,h)


def fusion_rectangle(list_rectangles):
    #obtention d'un top-left optimal(le plus en haut à gauche)
    #et obtention du pire-topleft (rectangle le plus en bas à droite)
    #on veut aussi la hauteur et largeur de ce pire rectangle
    if len(list_rectangles)!=0:
        x_tl_min = list_rectangles[0][0]
        y_tl_min = list_rectangles[0][1]
        x_tl_max = list_rectangles[0][0]
        y_tl_max = list_rectangles[0][1]
        w_temp,h_temp = list_rectangles[0][2], list_rectangles[0][3]
        
        for (x,y,w,h) in list_rectangles:
            x_tl_min = min(x_tl_min,x)
            y_tl_min = min(y_tl_min,y)
            if x >= x_tl_max:
                x_tl_max = x
                w_temp = w
            if y >= y_tl_max :
                y_tl_max = y
                h_temp = h
            y_max = max(y_tl_max,y)
        
        x_br_max, y_br_max = x_tl_max + w_temp, y_tl_max + h_temp
        
        w,h = ecart_hauteur_largeur(x_tl_min,y_tl_min,x_br_max, y_br_max)
        
        return (x_tl_min,y_tl_min,w,h)

    #si il n'y a pas de rectangle, on retourne un point
    return(0,0,0,0)
