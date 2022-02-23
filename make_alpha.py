import pickle
import requests # for alpha-binary digits

import matplotlib.pyplot as plt # for plotting

import numpy as np # to format images as arrays
import scipy.io # to convert .mat file into dictionary


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