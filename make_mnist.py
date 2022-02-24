import numpy as np
import pickle
import gzip

# import torch
# from torchvision import datasets, transforms # for MNIST


#################################################################################
############################### Using Torchvision ###############################
#################################################################################

# # Define a transform to normalize the data
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
# # Download and load the training data
# trainset = datasets.MNIST('../data/processed', download=False, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# # Download and load the test data
# testset = datasets.MNIST('../data/processed', download=False, train=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.shape, labels.shape) #loading batch of 64 images of shape 28x28

# plt.imshow(images[0].squeeze(), cmap='Greys_r')
# plt.show()

##############################################################################
############################### General method ###############################
##############################################################################

# add download with requests

def normalize(images, center=False, black_and_white=False):
    max = np.max(images)
    min = np.min(images)
    images_minmax = (images - min) / (max - min)
    
    if black_and_white:
        images_minmax = 0*(images_minmax == 0) + 1*(images_minmax > 0)
    if center:
        images_minmax = (images_minmax - .5) / .5
    
    return images_minmax

def one_hot(num_labels):
    one_hot_labels = []
    for label in num_labels:
        y = [0 for i in range(10)]
        y[int(label)] = 1
        one_hot_labels.append(y)
    one_hot_labels = np.array(one_hot_labels)
    return one_hot_labels

files = ['./data/raw/train-images-idx3-ubyte.gz', './data/raw/t10k-images-idx3-ubyte.gz',
         './data/raw/train-labels-idx1-ubyte.gz', './data/raw/t10k-labels-idx1-ubyte.gz']
files_desc = ['train_images', 'test_images',
              'train_labels', 'test_labels']
data = {}
for path, key in zip(files[:2], files_desc[:2]):
    with gzip.open(path, 'rb') as f:
        data[key] = (np.frombuffer(
                f.read(), np.uint8, offset=16
            ).reshape(-1, 28 * 28))
        
for path, key in zip(files[2:], files_desc[2:]):
    with gzip.open(path, 'rb') as f:
        data[key] = np.frombuffer(f.read(), np.uint8, offset=8)

# with open('./data/interim/mnist_numpy', 'wb') as f:
#     pickle.dump(data, f)
    
########################## Normalize ##########################

data[files_desc[0]] = normalize(data[files_desc[0]], black_and_white=True, center=False) # normalize train images
data[files_desc[1]] = normalize(data[files_desc[1]], black_and_white=True, center=False) # normalize test images
data[files_desc[2]] = one_hot(data[files_desc[2]]) # one hot train labels
data[files_desc[3]] = one_hot(data[files_desc[3]]) # one hot test labels

with open('./data/processed/mnist_numpy', 'wb') as f:
    pickle.dump(data, f)