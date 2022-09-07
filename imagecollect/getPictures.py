import flickrapi
import urllib
from PIL import Image

"""
This file has been used to dowload large datasets of images to train our cascade
"""

flickr = flickrapi.FlickrAPI(
    '8630790cff684af840d53c211648643e', '81760205a4268d22', cache=True)

N= 1000

#mot clé de la recherche
keyword = "plastic bottles"

#recherche
photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword,
                     extras='url_c', per_page=1000, sort='relevance')

#import des photos
for i, photo in enumerate(photos):
    url = photo.get('url_c')

    #test pour ne pas avoir d'erreur lors de la création du fichier
    if url != None:  
        name = "../datasetBottles/"+str(i)+'.jpg'
        urllib.request.urlretrieve(
            url, name)
    #nous voulons obtenir N images
    if i > N:
        break


    
