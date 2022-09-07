import numpy as np
import matplotlib.pyplot as plt


PATH_HISTORY="Data/history.npy"

#charge la liste correspondant au path (qui doit se finir par .npy)
def loadList(filename):
    tempNumpyArray=np.load(filename,allow_pickle=True)
    return tempNumpyArray.tolist()


history=loadList(PATH_HISTORY)


#graph des précisions en fonction du nombre d'epochs
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy');
    plt.show()


#graph des "losses" en fonction du nombres d'epoch
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.show()




if __name__ == '__main__':  
    plot_losses(history)
    plot_accuracies(history)