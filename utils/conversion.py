import os
import cv2
import sys

from image_processing_operations import resize_image, rgb_to_gray

# Le (relative) path qui mène au directory où sont situé les images
path = "../dataGrayResized/"

outputSize = (512, 384)


def grayConversion(basepath):
    # convertit en grayscale toutes les images (extensions jpg et png) du directory passé en argument.

    for entry in os.listdir(basepath):
        if entry.endswith(".png") or entry.endswith(".jpg"):
            pathEntry = basepath +"/"+ entry
            image = cv2.imread(pathEntry)
            gray = rgb_to_gray(image)
            cv2.imwrite(pathEntry, gray)


def resize(basepath):
    # redimensionne à 512x384 toutes les images du directory passé en argument (ratio 1.33:1) (éxécution en 20s)

    for entry in os.listdir(basepath):
        if entry.endswith(".png") or entry.endswith(".jpg"):
            pathEntry = basepath +"/" +entry
            image = cv2.imread(pathEntry)
            resizedImage = resize_image(image, outputSize)
            cv2.imwrite(pathEntry, resizedImage)



if __name__ == "__main__":
    for directory in ['n', 'p']:
        print(path+directory)
        resize(path+directory)
        grayConversion(path+directory)

