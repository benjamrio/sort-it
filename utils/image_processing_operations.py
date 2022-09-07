import cv2
import numpy as np


# En entrée une image sous forme cv2 et un tuple(w,h) indiquant les nouvelles dimensions
# En sortie l'image sous forme cv2 resized
def resize_image(img, new_dim):

    resized_image = cv2.resize(
        img, (new_dim[0], new_dim[1]), interpolation=cv2.INTER_CUBIC)
    return(resized_image)

# En entrée une image sous forme cv2 et l'angle de rotation que l'on souhaite appliquer
# En sortie l'image sous forme cv2 tournée


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1])
    return result

# En entrée une image sous forme cv2 et les paramètres du rectangle
# En sortie l'image concernée avec un rectangle dessiné dessus


def draw_rectangle(img, tlcorner, brcorner, color, line_thickness):
    # tl = top lef and br = bottom right
    img = cv2.rectangle(img, tlcorner, brcorner, color, line_thickness)
    return img


# En entrée une image sous forme cv2
# En sortie l'image convertie en grayScale
def rgb_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray)
