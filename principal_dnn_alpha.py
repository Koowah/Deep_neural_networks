import pickle
import numpy as np
import scipy.io # en attendant de dl torch sur pc portable
import matplotlib.pyplot as plt

from principal_rbm_alpha import RBM
from principal_dbn_alpha import DBN


######################################################################
#########################  HELPER FUNCTIONS  #########################
######################################################################

def one_hot(num_labels):
    one_hot_labels = []
    for label in num_labels:
        y = [0 for i in range(10)]
        y[label] = 1
        one_hot_labels.append(y)
    one_hot_labels = np.array(one_hot_labels)
    return one_hot_labels

# Helper function to get images formatted properly into array
def lire_alpha_digit(all_images, classes:list):
    data = []
    for cls in classes:
        for image in all_images[cls]:
            data.append(image.flatten().reshape(-1,1))
    
    return np.array(data)

# DNN helper functions
def sigmoid(x, derivate=False):
    sigm = 1 / (1 + np.exp(-x))
    if derivate:
        return sigm * (1 - sigm)
    return sigm

def calcul_softmax(scores):
    exps = np.exp(scores)
    softmax = exps / exps.sum()
    
    assert softmax.shape == scores.shape, 'probability layer should have the same shape as hidden layer'
    assert round(softmax.sum(), 2) == 1.0, 'probabilities should sum to 1'
    
    return softmax

def cross_entropy(y, y_pred, derivate=False):
    return - (y_pred - y) if derivate else - (y * np.log(y_pred))

#####################################################################
###############################  DNN  ###############################
#####################################################################

class DNN(DBN): # we consider a deep neural network as a deep belief network with one last layer to which we apply softmax - allows us to pretrain !
    def __init__(self, n_v=None, layers=None, k=1, dic_load=None, dnn_load=None):
        # Not the cleanest way to define the NN structure wrt the underlying DBN structure
        # as DBN isn't updated by training here - only W and B are
        # but allows for consistent notations and possibility
        # to load pretrained DBN & trained DNN separately
        if dnn_load is None:
            super().__init__(n_v, layers, k, dic_load)
            # we copy our DBN structure
            self.weights = [rbm.W for rbm in self.rbms]
            self.biases = [rbm.c for rbm in self.rbms]
        else:
            # all we need to make predictions
            self.n_layer = dnn_load['n_layer']
            self.weights = dnn_load['weights']
            self.biases = dnn_load['biases']
            
            assert self.n_layer == len(self.weights), "n_layer inconsistent with number of weight matrices"
            assert self.n_layer == len(self.biases), "n_layer inconsistent with number of bias vectors"
                    
    def forward(self, X):
        return super().forward(X)
    
    def forward_DNN(self, X):
        n_samples = X.shape[0]
        X = X.T
        
        w = self.weights
        b = self.biases
        
        for i in range(self.n_layer):
            X = sigmoid(np.dot(w[i], X) + b[i])
        
        assert X.shape == (self.biases[-1].shape[0], n_samples), 'problem forward_DNN activation shape'
        
        return X
    
    def backward(self, H):
        return super().backward(H)
    
    def pretrain_model(self, X, epochs=1, learning=0.01, save=False):
        return super().pretrain_model(X, epochs, learning, save)
    
    # DNN specific functions
    def predict(self, X):
        return np.argmax(calcul_softmax(self.forward_DNN(X)))
                
    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # forward pass - adapted notations to literature
        activation = x.reshape(-1,1)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        i = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            
            # assert not((activation == self.rbms[i].forward(activations[i].T)[0].T).any() == False), "activation isn't right"
            activations.append(activation)
            i += 1
            
        # backward pass
        delta = cross_entropy(activations[-1], y.reshape(-1,1), derivate=True) * \
            sigmoid(zs[-1], derivate=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.n_layer):
            z = zs[-l]
            sp = sigmoid(z, derivate=True)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)
    
    def update_batch(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta)*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    
    def train_model(self, X, y, batch_size=10, epochs=300, learning_rate=0.01, save=False):
        num_batches = X.shape[0]//batch_size + (X.shape[0] % batch_size > 0)
        data = list(zip(X, y))
        
        for epoch in range(epochs):
            np.random.shuffle(data)
            batches = [data[i * batch_size : min((i+1)*batch_size, X.shape[0]-1)] for i in range(num_batches)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if epoch % 10 == 0:
                y_pred = np.array(list(map(calcul_softmax, self.forward_DNN(X).T)))
                labels = np.array(list(map(np.argmax,y)))
                pred_labels = np.array(list(map(np.argmax, y_pred)))
                print(f'Epoch {epoch} - Cross-entropy : {cross_entropy(y, y_pred).mean()} - Train accuracy : {(labels == pred_labels).mean():.2%}')
        return
    
    def save_model(self, name:str, dnn=False):
        if dnn: # save the trained DNN
            dic = {'weights':self.weights, 'biases':self.biases, 'n_hs':self.n_hs, 'n_layer':self.n_layer, 'rbms':self.rbms}
            with open(f'{name}.txt', 'wb') as f:
                pickle.dump(dic, f)
        else: # save the pretrained DBN
            return super().save_model(name)
    
    @classmethod
    def load_model(cls, path:str, dnn=False):
        if dnn:
            with open(path, 'rb') as tbon:
                dic = pickle.load(tbon)
            return cls(dnn_load=dic) # dnn_load instead of dic_load ( which should be renamed dbn_load)
        else:
            return super().load_model(path)

    
    
############################################################################
############################### MAIN PROGRAM ###############################
############################################################################
  
def main(pretrain=False, load=False, train=True):
    ############################### WARNING ###############################
    np.random.seed(42) # set random seed for reproducibility
    
    # Better to pretrain without setting random seed then train, as contrastive divergence
    # relays on gibbs sampling and therefore on unpredictability of sampling
    
    
    ############################### Prepare DATA ###############################
    # Convert from .mat to usable arrays and plot
    file_mat = './data/raw/alpha_binary.mat'
    mat = scipy.io.loadmat(file_mat)
    
    print('Type scipy loaded .mat : ', type(mat))
    print('mat dictionary keys : ', mat.keys())
    print("mat['dat'] values shape : ", mat['dat'].shape) # 39 samples for each of the 36 classes - 10 digits & 26 letters
    print("mat['classlabels'] elements : ", mat['classlabels'][0])

    ###################################
    images = mat['dat'] # OUR MAIN DATA
    labels = np.array([label.item() for label in mat['classlabels'][0]]) # retrieves the labels
    
    num_labels = np.array([int(label) for label in labels[:10]])
    one_hot_labels = one_hot(num_labels)
    ###################################

    plt.imshow(images[3][0], cmap='Greys_r')
    plt.show()

    plt.imshow(images[10][0], cmap='Greys_r')
    plt.show()

    data = lire_alpha_digit(images, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # retrieve all numerical classes
    
    labels_array = []
    for index, classe in enumerate(images):
        for image in classe:
            labels_array.append(one_hot_labels[index])
        if index==9:
            break

    labels_array = np.array(labels_array)
    
    
    ###############################  Pretrain DNN  ###############################
    n_v = 20*16
    h_1 = 20*10
    h_2 = 20*10
    h_3 = 20*8
    output = 10
    
    if pretrain:
        # pretrain or load pretrained dbn
        if load:
            dnn = DNN.load_model('./models/DBN_pretrain_alpha_3_150.txt') # load our trained DBN
            
            # add last layer to make DNN
            dnn.weights = [rbm.W for rbm in dnn.rbms]
            dnn.weights.append(np.random.normal(0, .1, size=(output, dnn.n_hs[dnn.n_layer])))
            dnn.biases = [rbm.c for rbm in dnn.rbms]
            dnn.biases.append(np.zeros(output).reshape(-1,1))
            dnn.n_layer += 1
        else:
            layers = [
                (h_1, None),
                (h_2, None),
                (h_3, None),
                # (output, None), # yields much worse results for those who wondered
            ]
            dnn = DNN(n_v, layers) # initialize DBN
            dnn.pretrain_model(data, epochs=150, save=False) # train DBN greedily
            
            # add last layer to make our DNN
            dnn.weights = [rbm.W for rbm in dnn.rbms]
            dnn.weights.append(np.random.normal(0, .1, size=(output, dnn.n_hs[dnn.n_layer - 1])))
            dnn.biases = [rbm.c for rbm in dnn.rbms]
            dnn.biases.append(np.zeros(output).reshape(-1,1))
            dnn.n_layer += 1
    
    
    ###############################  Initialize Random DNN  ###############################    
    else:
        if load:
            dnn = DNN.load_model('./models/dnn_trained_alpha.txt', dnn=True)
        else:
            layers = [
                (h_1, None),
                (h_2, None),
                (h_3, None),
                (output, None),
            ]
            dnn = DNN(n_v, layers)
        
    X = data.copy().squeeze()
    y = labels_array.copy()
    
    # Here train-test split
    
    ###############################  Train DNN  ###############################
    if train:
        dnn.train_model(X, y)
        dnn.save_model('./models/dnn_trained_alpha.txt', dnn=True)
        
    ###############################  Test DNN  ###############################
    else:
        # Utilize test set to assess accuracy
        pass
    

if __name__ == '__main__':
    main(pretrain=True, load=True)
    
    # def numpy uniform permutation/sklearn -> train test split to data and labels_array -> check distribution -> def accuracy -> make plots -> apply to MNIST -> DONE