import numpy as np
import scipy.io # en attendant de dl torch sur pc portable
import matplotlib.pyplot as plt

from principal_rbm_alpha import RBM
from principal_dbn_alpha import DBN


# Helper function to get images formatted properly into array
def lire_alpha_digit(all_images, classes:list):
    data = []
    for cls in classes:
        for image in all_images[cls]:
            data.append(image.flatten().reshape(-1,1))
    
    return np.array(data)


#####################################################################
###############################  DNN  ###############################
#####################################################################

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
    return - (y / y_pred) if derivate else - (y * np.log(y_pred))

class DNN(DBN): # we consider a deep neural network as a deep belief network with one last layer to which we apply softmax - allows us to pretrain !
    def __init__(self, n_v=None, layers=None, k=1, dic_load=None):
        super().__init__(n_v, layers, k, dic_load)
        self.weights = [rbm.W for rbm in self.rbms]
        self.biases = [rbm.b for rbm in self.rbms]
            
    def forward(self, X):
        return super().forward(X)
    
    def backward(self, H):
        return super().backward(H)
    
    def pretrain_model(self, X, epochs=1, learning=0.01, save=False):
        return super().pretrain_model(X, epochs, learning, save)
    
    # DNN specific functions
    def predict(self, X):
        return calcul_softmax(self.forward(X))
            
    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # forward pass - adapted notations to literature
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            
            assert activation == self.forward(x.T), "activation isn't right"
            activations.append(activation)
            
        # backward pass
        delta = cross_entropy(activations[-1], y, derivate=True) * \
            sigmoid(zs[-1], derivate=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.n_layers):
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
        self.weights = [w-(eta/len(batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb 
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
                print(f'Epoch {epoch} - Cross-entropy : {cross_entropy(y, self.forward(X.T))}')
        return
                
    
    def save_model(self, name):
        return super().save_model(name)
    
    @classmethod
    def load_model(cls, path: str):
        return super().load_model(path)

    
    
############################################################################
############################### MAIN PROGRAM ###############################
############################################################################
  
def main(pretrain=False, load=False, train=True):
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
    ###################################

    plt.imshow(images[3][0], cmap='Greys_r')
    plt.show()

    plt.imshow(images[10][0], cmap='Greys_r')
    plt.show()

    data = lire_alpha_digit(images, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # retrieve all numerical classes
    
    labels_array = []
    for index, classe in enumerate(images):
        for image in classe:
            labels_array.append(labels[index])

    labels_array = np.array(labels_array)
    # train_data = list(zip(data, labels_array[:39*9])) # zips data and relevant label in tuples
    
    ###############################  Pretrain DNN  ###############################
    n_v = 20*16
    h_1 = 20*10
    h_2 = 20*10
    h_3 = 20*8
    output = 10
    
    if pretrain:
        # pretrain or load pretrained dbn
        if load:
            dnn = DNN.load_model('./models/DNN_pretrained_4_150.txt')
        # else:
            # layers = [
            #     (h_1, None),
            #     (h_2, None),
            #     (h_3, None),
            #     (output, None),
            # ]
            # dnn = DNN(n_v, layers)
            # dnn.pretrain_model(data, epoch=150)
        
    else:
        layers = [
            (h_1, None),
            (h_2, None),
            (h_3, None),
            (output, None),
        ]
        dnn = DNN(n_v, layers)
        
    X = data.copy()
    y = labels_array.copy()
    dnn.train_model(X, y)
    dnn.save_model('./models/dnn_trained')
    

if __name__ == '__main__':
    main()