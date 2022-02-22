import pickle
import requests # for alpha-binary digits

import torch
from torchvision import datasets, transforms # for MNIST

import matplotlib.pyplot as plt # for plotting

import numpy as np # to format images as arrays
import scipy.io # to convert .mat file into dictionary

#####################################################################
############################### MNIST ###############################
#####################################################################

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.MNIST('../data/processed', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Download and load the test data
testset = datasets.MNIST('../data/processed', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape, labels.shape) #loading batch of 64 images of shape 28x28

plt.imshow(images[0].squeeze(), cmap='Greys_r')
plt.show()


#####################################################################
############################### ALPHA ###############################
#####################################################################

# Download and save alpha_binary from website
alpha_binary_mat = requests.get('https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat')

with open('../data/raw/alpha_binary.mat', 'wb') as f:
    f.write(alpha_binary_mat.content)

# Convert from .mat to usable arrays

file_mat = '../data/raw/alpha_binary.mat'
mat = scipy.io.loadmat(file_mat)
print('Type scipy loaded .mat : ', type(mat))

print('mat dictionary keys : ', mat.keys())
print("mat['dat'] values shape : ", mat['dat'].shape) # 39 samples for each of the 36 classes - 10 digits & 26 letters
print("mat['classlabels'] elements : ", mat['classlabels'][0])

labels = np.array([label.item() for label in mat['classlabels'][0]])
images = mat['dat']
alpha_binary_processed = {'images':images, 'labels':labels}

with open('./data/processed/alpha_binary_processed', 'wb') as f:
    pickle.dump(alpha_binary_processed, f)

plt.imshow(images[3][0], cmap='Greys_r')
plt.show()

plt.imshow(images[10][0], cmap='Greys_r')
plt.show()