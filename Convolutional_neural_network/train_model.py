def run():    
    import os
    import torch
    import torchvision
    from torch.utils.data import random_split
    import torchvision.models as models
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms
    from torch.utils.data.dataloader import DataLoader
    import numpy as np
    from create_model import create_empty_model,get_default_device,to_device, accuracy
    
    #pour éviter des erreurs éventuelles
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    

    """
    Déclaration des constantes
    """
    #directory de la base d'images
    DATA_DIR= 'Data/Garbage_classification'
    
    #definition de la taille des batchs (sous ensembles des dataLoaders (dataset transfomés))
    BATCH_SIZE= 32

    #path où l'on sauve l'historique sous forme .npy
    PATH_HISTORY="Data/history.npy"

    #paramètres d'entrainements
    NUM_EPOCHS = 5
    OPT_FUNC = torch.optim.Adam
    LR = 5.5e-5



    """
    Déclaration des fonctions et des classes
    """

    '''
    En entrée : un path qui indique le dossier contenant la base d'images
    En sortie: Un ImageFolder, contenant cette banque d'images traitées (resized et transformées en tensor, modules de Pytorch), pret à être utilisé par un model
    '''
    def tranform_dataset(data_directory):
        transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        dataset = ImageFolder(data_directory, transform = transformations) 
        return (dataset)
    
    
    
    '''
    En entrée: un ImageFolder
    En sortie: trois dataset (training, validation et test) de taille fixées contenant des images aléatoirement choisis dans notre banque d'image 
    '''
    def generate_train_val_test_datasets(data):
        #on separe la data set en train set, valuation set et test set à l'aide d'une répartition aléatoire
        random_seed = 12
        torch.manual_seed(random_seed)
        train_ds , val_ds , test_ds = random_split(data, [1593, 176, 758])
        return (train_ds,val_ds,test_ds)
    
    

    '''
    En entrée: deux datasets (training et validation) ainsi que la taille désirée du DataLoader training
    En sortie: deux DataLoader (training et validation) avec des 
    '''
    def gen_data_load(train_ds, val_ds,batch_size):
        train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True) #la taille des batchs de validation est le double de celle des batches de training
        return(train_dl,val_dl)



    #on crée une nouvelle classe pour adapter les dataLoaders au device utilisé
    class DeviceDataLoader():
        #prépare un Dataloader à être envoyé sur le device utilisé
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            #produit un batch après le charger sur le device
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            #nombre de batchs
            return len(self.dl)

    

    #preparation pour le training
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
   


    '''
    En entrée: le nombre d'epoch, le learning rate, le model, les datasets training et validation de type DeviceDataLoader et la fonction d'optimisation
    En sortie: un arry représentant l'historique (a la i-eme case y figure) 
    '''
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)

        for epoch in range(epochs):
            
            # Training Phase 
            model.train()
            train_losses = []

            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)

            history.append(result)
        return history



    '''
    En entrée: une liste et un path qui se termine par .npy
    Cette fonction sauve la liste dans un fichier .npy 
    '''
    def saveList(myList,filename):
        np.save(filename,myList)
        print("Saved successfully!")

    
    
    """
    Execution de l'entrainement
    """
    #initialisation du dataset au bon format
    dataset=transform_dataset(DATA_DIR)

    #inititalisation des datasets training, validation et test
    train_ds,val_ds,test_ds = generate_train_val_test_datasets(dataset)
   
    #initialisation des dataLoaders training et validation
    train_dl,val_dl = gen_data_load(train_ds,val_ds,BATCH_SIZE)

    #On crée un model grâce au fichier create_model (construit l'architecture du réseau)
    model = create_empty_model()

    #on recupère le device à utliser
    device = get_default_device()
    
    #instantiation des DeviceDataLoaders
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    
    @torch.no_grad()
    
    #préparation au training
    evaluate(model, val_dl)
    
    #training du modèle, et on récupère l'historique de précision et des loss en fonction du numéro de l'epoch
    history = fit(NUM_EPOCHS, LR, model, train_dl, val_dl, OPT_FUNC)

    #Enfin, on sauve nos paramètres variables de notre modèle...
    torch.save(model.state_dict(), "Data/model_test_2.pt")
    
    #... ainsi que l'historique
    saveList(history,PATH_HISTORY)



if __name__ == '__main__':
    run()
