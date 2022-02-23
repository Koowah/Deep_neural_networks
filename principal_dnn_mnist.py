import time
import pickle

import torch
from torchvision import datasets, transforms

import numpy as np
import scipy.io # en attendant de dl torch sur pc portable
import matplotlib.pyplot as plt

from principal_rbm_alpha import RBM
from principal_dbn_alpha import DBN


######################################################################
#########################  HELPER FUNCTIONS  #########################
######################################################################

# one_hot encodes numerical labels
def one_hot(num_labels):
    one_hot_labels = []
    for label in num_labels:
        y = [0 for i in range(10)]
        y[int(label)] = 1
        one_hot_labels.append(y)
    one_hot_labels = np.array(one_hot_labels)
    return one_hot_labels

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

def accuracy(test_data, test_labels, dnn):
    X_test = test_data.squeeze()
    y_test = test_labels
        
    # predict labels
    y_pred = np.array(list(map(calcul_softmax, dnn.forward_DNN(X_test).T)))
    pred_labels = np.array(list(map(np.argmax, y_pred))).reshape(-1,1)
        
    # accuracy
    accuracy = (y_test == pred_labels).mean()
    
    return accuracy

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
    
    def pretrain_model(self, X, batch_size=10, epochs=100, learning=0.01, save=False):
        return super().pretrain_model(X, batch_size, epochs, learning, save)
    
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
            # t0 = time.time()
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            # t1 = time.time()
            # print('backpropagate for one input time : ', t1 - t0)
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
            # t0 = time.time()
            np.random.shuffle(data)
            batches = [data[i * batch_size : min((i+1)*batch_size, X.shape[0]-1)] for i in range(num_batches)]
            # t1 = time.time()
            # print('shuffle + make batches time : ', t1 - t0)
            for batch in batches:
                # t2 = time.time()
                self.update_batch(batch, learning_rate)
                # t3 = time.time()
                # print('update_batch time : ', t3 - t2)
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
    t0 = time.time()
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    if train:
        trainset = datasets.MNIST('./data/processed', download=False, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset)) # can specify batch_size and shuffle=True but will be done manually for the sake of the exercise
    # Download and load the test data
    testset = datasets.MNIST('./data/processed', download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset)) # same remark
    
    #######################################################
    ###################### MAIN DATA ######################
    # Format images as numpy ndarray of shape (n_sample, 1, 28, 28) 
    # and labels as array of shape (n_sample)
    # n_sample = 60 000 for train 10 000 for test
    if train:
        train_data = next(iter(trainloader))[0].numpy().reshape(60_000, -1, 1) # reshaping consistent with dbn pretraining - could be worked on to be more "natural"
        train_labels = next(iter(trainloader))[1].numpy().reshape(-1,1)
        one_hot_train_labels = one_hot(train_labels)
        
    test_data = next(iter(testloader))[0].numpy().reshape(10_000, -1, 1)
    test_labels = next(iter(testloader))[1].numpy().reshape(-1,1)
    # one_hot_test_labels = one_hot(test_labels)
    #######################################################
    #######################################################
    t1 = time.time()
    print('Loading data and formatting time : ', t1 - t0)
    plt.imshow(test_data[0].reshape(28,28), cmap='Greys_r')
    plt.show()

    
    ###############################  Pretrain DNN  ###############################
    n_v = 784
    h_1 = 128
    h_2 = 64
    output = 10
    
    if pretrain:
        # pretrain or load pretrained dbn
        if load:
            dnn = DNN.load_model('./models/DBN_pretrain_mnist_3_150.txt') # load our trained DBN
            
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
                # (h_3, None),
                # (output, None), # yields much worse results for those who wondered
            ]
            dnn = DNN(n_v, layers) # initialize DBN
            dnn.pretrain_model(test_data, batch_size=100, epochs=150, save=True) # train DBN greedily
            
            # add last layer to make our DNN
            dnn.weights = [rbm.W for rbm in dnn.rbms]
            dnn.weights.append(np.random.normal(0, .1, size=(output, dnn.n_hs[dnn.n_layer - 1])))
            dnn.biases = [rbm.c for rbm in dnn.rbms]
            dnn.biases.append(np.zeros(output).reshape(-1,1))
            dnn.n_layer += 1
    
    
    ###############################  Initialize Random DNN  ###############################    
    else:
        if load:
            dnn = DNN.load_model('./models/dnn_trained_mnist.txt', dnn=True)
        else:
            layers = [
                (h_1, None),
                (h_2, None),
                # (h_3, None),
                (output, None),
            ]
            dnn = DNN(n_v, layers)
        
    
    ###############################  Train DNN  ###############################
    if train:
        X_train = train_data.squeeze()
        y_train = one_hot_train_labels
        
        dnn.train_model(X_train, y_train, batch_size=64, epochs=200, learning_rate=.01)
        dnn.save_model('./models/dnn_trained_mnist', dnn=True)
        
    ###############################  Test DNN  ###############################
    else:
        # Utilize test set to assess accuracy
        acc = accuracy(test_data, test_labels, dnn)
        print(f'Accuracy on test set : {acc:.2%}')
    

if __name__ == '__main__':
    main(pretrain=False, load=True, train=False)
    
    # def numpy uniform permutation/sklearn -> train test split to data and labels_array -> check distribution -> def accuracy -> make plots -> apply to MNIST -> DONE