import torch
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
#puisque l'on éxécute via main qui est frere d'un parent
from convolutional_neural_network.create_model import *


#La ligne de code suivant permet d'eviter une erreur qui apparait sur certaines configurations
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#Composition de resize puis d'une conversion en tensor
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


#On charge le squelette (l'architectur)
model = create_empty_model()


#On load le model(les poids et les variables) sur ce modèle
model1 = model.load_state_dict(torch.load("Data/model_final_3_epoch.pt"))


#sortie du mode training (désactivation de certains layers)
model.eval()


'''
En entrée : un des six string output du réseau (correspondant au matériel)
En sortie : la poubelle dans laquelle doit aller ce matériau
'''
def triPoubelle(material: str):
    dico_tri = {'plastic': 'yellow', 
            'cardboard': 'yellow or blue', 
            'paper': 'blue',
            'glass': 'green',
            'metal': 'red',
            'trash': 'red'}
    return dico_tri[material]



'''
En entrée: une image (type Image) et un model pytorch
En sortie: La classification prédite par notre model de cette image
'''
def predict_image(img, model):
    # Conversion au bon format pour l'input
    xb = to_device(img.unsqueeze(0), device)
    # On obitent les prédictions via le modèle
    yb = model(xb)
    # On choisit l'output avec la probabilité la plus élevée
    prob, preds  = torch.max(yb, dim=1)
    # on récupère le label correspondant à ce label
    return classes[preds[0].item()]




#predit le label d'une image (via le path en argument) en rentournant le couple (matériel, poubelle)
def predict_path(path):
    image = Image.open(Path(path))
    example_image = transformations(image)
    print("Image ready to be predicted...")
    materiel = predict_image(example_image,model)
    poubelle = triPoubelle(materiel)
    return((materiel,poubelle))




if __name__=="__main__":
    print(predict_path("Data/test_image.jpg"))