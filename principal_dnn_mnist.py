import numpy as np
import scipy.io # en attendant de dl torch sur pc portable
import matplotlib.pyplot as plt

from principal_rbm_alpha import RBM
from principal_dbn_alpha import DBN


# Useful function to get images formatted properly into array
def lire_alpha_digit(all_images, classes:list):
    data = []
    for cls in classes:
        for image in all_images[cls]:
            data.append(image.flatten().reshape(-1,1))
    
    return np.array(data)


#####################################################################
###############################  RBM  ###############################
#####################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calcul_softmax(rbm:RBM, data):
    h, _ = rbm.forward(data.T)
    sum_exp = np.exp(h).sum()
    softmax = np.exp(h) / sum_exp
    
    assert softmax.shape == h.shape, 'probability layer should have the same shape as hidden layer'
    assert round(softmax.sum(), 2) == 1.0, 'probabilities should sum to 1'
    
    return softmax

class DNN(DBN): # we consider a deep neural network as a deep belief network with one last layer to which we apply softmax - allows us to pretrain !
    def __init__(self, n_v, layers, k=1):
        super().__init__(n_v, layers, k) # inherit from DBN
        # how to inherit DBN and add one last layer ? gotta change layers in initialization and that's it
    
    
############################################################################
############################### MAIN PROGRAM ###############################
############################################################################
  
def main():
    ############################### Prepare DATA ###############################
    # Convert from .mat to usable arrays and plot
    file_mat = './data/raw/alpha_binary.mat'
    mat = scipy.io.loadmat(file_mat)
    
    print('Type scipy loaded .mat : ', type(mat))
    print('mat dictionary keys : ', mat.keys())
    print("mat['dat'] values shape : ", mat['dat'].shape) # 39 samples for each of the 36 classes - 10 digits & 26 letters
    print("mat['classlabels'] elements : ", mat['classlabels'][0])

    # labels = np.array([label.item() for label in mat['classlabels'][0]]) # retrieves the labels
    ###################################
    images = mat['dat'] # OUR MAIN DATA
    ###################################

    plt.imshow(images[3][0], cmap='Greys_r')
    plt.show()

    plt.imshow(images[10][0], cmap='Greys_r')
    plt.show()

    data = lire_alpha_digit(images, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # retrieve all numerical classes

    

if __name__ == '__main__':
    main()