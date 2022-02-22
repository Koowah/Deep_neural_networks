import pickle

import matplotlib.pyplot as plt # for plotting
import numpy as np # to format images as arrays
import scipy.io # to convert .mat file into dictionary


# Helper function to get images formatted properly into array
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

class RBM:
    
    def __init__(self, n_v, n_h, W=None, b=None, c=None, k=1): # k=1 -> only one gibbs iterations -> CD1
        assert n_v != 0 and n_h != 0
        self.n_v = n_v
        self.n_h = n_h
        shape = (n_h, n_v)
        
        self.W = W if W is not None else np.random.uniform(-1, 1, size=shape)
        self.b = b if b is not None else np.zeros(n_v).reshape(-1,1)
        self.c = c if c is not None else np.zeros(n_h).reshape(-1,1)

        assert self.W.shape==shape and n_v == len(self.b) and n_h == len(self.c)
        
        self.k = k
        return
        
    def forward(self, V): # entree_sortie
        n_sample, n_v = V.shape
        
        hsignal = np.dot(V, self.W.T) + self.c.T
        assert hsignal.shape == (n_sample, self.n_h)
        Hp = sigmoid(hsignal)
        
        #s = np.random.uniform(0, 1, size=hsignal.shape)
        #Hs = (s < Hp) * 1  # same as:
        Hs = np.random.binomial(1, Hp, size=Hp.shape)
        return Hp, Hs
    
    def backward(self, H): # sortie_entree
        n_sample, n_h = H.shape
        
        vsignal = np.dot(H, self.W) + self.b.T
        assert vsignal.shape == (n_sample, self.n_v)
        #print(vsignal)
        Vp = sigmoid(vsignal)
        
        s = np.random.uniform(0, 1, size=vsignal.shape)
        Vs = (s < Vp) * 1
        return Vp, Vs

    def gibbs(self, V):  #return (probability, samples) of visible units
        Vs = V
        for i in range(self.k):
            Hp, Hs = self.forward(Vs)
            Vp, Vs = self.backward(Hs)
            
        return Hp, Hs, Vp, Vs
    
    def contrastive_divergence(self, V, learning=0.01):
        #set_trace()
        n_sample, n_v = V.shape
        
        Vs = V
        Hp, Hs, Vp_, Vs_ = self.gibbs(Vs)   # underscore _ refers to tilde for negative sample
        Hp_, Hs_ = self.forward(Vs_)

        Vs1 = np.mean(Vs, axis=0) 
        Vs2 = np.mean(Vs_, axis=0) 
        Hp1 = np.mean(Hp, axis=0)
        Hp2 = np.mean(Hp_, axis=0)
        
        # note, there are variances in how to compute the gradients.
        
        Eh_b = Vs1; Evh_b = Vs2      # Evh_b refers to the Expectation (over v and h) of -logP(v) gradient wrt b
        
        Eh_c = Hp1; Evh_c = Hp2 

        g_b = (Evh_b - Eh_b).reshape(-1,1)  # gradient of -logP(v) wrt b
        g_c = (Evh_c - Eh_c).reshape(-1,1)

        Eh_W = np.outer(Eh_c, Eh_b)
        Evh_W = np.outer(Evh_c, Evh_b)
        g_W = Evh_W - Eh_W
    
        self.W -= g_W * learning
        self.b -= g_b * learning
        self.c -= g_c * learning        
        return
    
    def reconstruct(self, V):
        Hp, Hs = self.forward(V)
        Vp, Vs = self.backward(Hp)
        return Vp, Vs
    
    def train_rbm(self, train_data, n_epoch=100, learning=0.01): # train_rbm
        for _ in range(n_epoch):
            MSE = []
            data = train_data.copy()
            np.random.shuffle(data)
            for x in data:
                self.contrastive_divergence(x.T, learning)
                reconstructed = self.reconstruct(x.T)
                MSE.append(((x.T - reconstructed)**2).sum()/len(x))
            print(f'MSE for epoch {_}: {np.array(MSE).mean()}')                      
        return
    
    def generate_image(self, iter_gibbs, number_image): # generer_image_rbm
        images = []
        for _ in range(number_image):
            v = np.zeros(self.n_v).reshape(1, -1)
            for _ in range(iter_gibbs):
                _, v = self.reconstruct(v)
            images.append(v)
        return images
    
    def save_model(self, name):
        dic = {'n_v':self.n_v, 'n_h':self.n_h, 'W':self.W, 'b':self.b, 'c':self.c}
        with open(f'{name}.txt', 'wb') as f:
            pickle.dump(dic, f)
    
    @classmethod
    def load_model(cls, path:str):
        with open(path, 'rb') as tbon:
            dic = pickle.load(tbon)
        return cls(n_v=dic['n_v'], n_h=dic['n_h'], W=dic['W'], b=dic['b'], c=dic['c'])

    
    
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

    ###############################  RBM Generative Power  ###############################
    rbm = RBM(20*16, 20*8) # define RBM structure
    data = lire_alpha_digit(images, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # retrieve all numerical classes

    # Uncomment if we want to retrieve all labels and zip them with relevant images
    # useful for class specific generation - gives much better results !

    # labels_array = []
    # for index, classe in enumerate(images):
    #     for image in classe:
    #         labels_array.append(labels[index])

    # labels_array = np.array(labels_array)
    # train_data = list(zip(data, labels_array[:39*3]))

    rbm.train_rbm(data, n_epoch=150) # training our model with 300 iterations of gradien descent

    plt.imshow(data[39*2].reshape(20,16), cmap='Greys_r') # data example
    plt.show()

    plt.imshow(rbm.reconstruct(data[39*2].T)[0].reshape(20,16), cmap='Greys_r') # RBM reconstructed data
    plt.show()

    generated_images = rbm.generate_image(iter_gibbs=1000, number_image=80) # generating 100 images from 1000 gibbs iterations each - 1000 is kind of overkill

    # plotting all generated images in a table
    fig = plt.figure(figsize=(16, 64))
    columns = 5
    rows = 16
    for i in range(1, columns*rows +1):
        img = generated_images[i-1].reshape(20,16)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='Greys_r')
    plt.show()
    

if __name__ == '__main__':
    main()