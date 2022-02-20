import matplotlib.pyplot as plt # for plotting
import numpy as np # to format images as arrays
import scipy.io # to convert .mat file into dictionary

from principal_rbm_alpha import RBM # retrieve RBM class from relevant file


# Useful function to get images formatted properly into array
def lire_alpha_digit(all_images, classes:list):
    data = []
    for cls in classes:
        for image in all_images[cls]:
            data.append(image.flatten().reshape(-1,1))
    
    return np.array(data)


#####################################################################
###############################  DBN  ###############################
#####################################################################

class DBN():
    def __init__(self, n_v, layers, k=1):
        
        if n_v is None or layers is None: raise ValueError("Incorrect inputs for layer 0.")
        
        n_hs = [n_v]        
        n_layer = 0
        
        rbms = []
        for (n_h, model) in layers:
            n_layer += 1
            if n_h <= 0: raise ValueError("Incorrect inputs for layer %d" % (n_layer))
            else: n_hs.append(n_h)

            rbm = RBM(n_hs[n_layer-1], n_h, k=k)
            rbms.append(rbm)

        self.n_hs = n_hs
        self.n_layer = n_layer
        self.rbms = rbms
        return
    
    def forward(self, X):
        
        Hp = X
        for i in range(self.n_layer):
            Hp, Hs = self.rbms[i].forward(Hp)
        
        return Hp, Hs

    def backward(self, H):

        Vp = H
        for i in reversed(range(self.n_layer)):
            Vp, Vs = self.rbms[i].backward(Vp)
        
        return Vp, Vs

    def train_model(self, X, epochs=1, learning=0.01):
        
        for layer in range(self.n_layer):
            print(f'Layer {layer + 1} training :')
            self.rbms[layer].train_model(X, n_epoch=epochs, learning=learning)
            X = np.swapaxes((np.array(list(map(lambda x : self.rbms[layer].forward(x.T), X)))[:, 0, :, :]), 1, 2)
        
        return
    
    def reconstruct(self, X):
        h_layer = self.n_layer - 1
        Hp = X
        for i in range(h_layer):
            Hp, Hs = self.rbms[i].forward(Hp)
        
        Vp, Vs = self.rbms[h_layer].reconstruct(Hp)

        for i in reversed(range(h_layer)):
            Hp, Hs = self.rbms[i].backward(Hp)

        return Hp, Hs
    
    def generate_image(self, iter_gibbs, n_images):
        images = []
        for _ in range(n_images):
            v = np.zeros(self.n_hs[0]).reshape(1, -1) # n_hs[0] = n_v first layer
            for _ in range(iter_gibbs):
                _, v = self.reconstruct(v)
            images.append(v)
        return images


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
    
    
    ###############################  DBN Generative Power  ###############################
    # DBN architecture
    n_v = 20*16 # visible units
    h_1 = 20*10 # hidden layer 1
    h_2 = 20*10  # hidden layer 2
    h_3 = 20*8

    # properly formatting layers for our __init__ function
    layers = [
        (h_1, None),
        (h_2, None),
        (h_3, None),
    ]

    dbn = DBN(n_v, layers) # instanciate dbn with above structure
    data = lire_alpha_digit(images, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # retrieve all numerical classes
    
    # plot a number
    plt.imshow(data[39*2].T.reshape(20,16), cmap='Greys_r')
    plt.title('alpha_digit 2')
    plt.show()
    
    # reconstruct before training
    plt.imshow(dbn.reconstruct(data[39*2].T)[0].reshape(20,16), cmap='Greys_r')
    plt.title('reconstruct before training')
    plt.show()
    
    dbn.train_model(data, epochs=150) # train dbn
    
    # reconstruct after training
    plt.imshow(dbn.reconstruct(data[39*2].T)[0].reshape(20,16), cmap='Greys_r')
    plt.title('reconstruct after training')
    plt.show()
    
    # generate images and plot in figure
    generated_images_dbn = dbn.generate_image(1000, 80) # 80 images from 1000 gibbs iterations each - 1000 is definetly overkill
    
    fig = plt.figure(figsize=(16, 48))
    plt.title('Trained DBN generated images')
    columns = 5
    rows = 16
    for i in range(1, columns*rows +1):
        img = generated_images_dbn[i-1].reshape(20,16)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    main()