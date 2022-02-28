import numpy as np
import requests
import pickle
import gzip
import os

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

########################## Helper functions ##########################
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

########################## Main Program ##########################
def main(download=False, num_train=60_000):
    files_paths = ['./data/raw/train-images-idx3-ubyte.gz', './data/raw/t10k-images-idx3-ubyte.gz',
            './data/raw/train-labels-idx1-ubyte.gz', './data/raw/t10k-labels-idx1-ubyte.gz']
    files_desc = ['train_images', 'test_images',
                'train_labels', 'test_labels']
    data = {}
    
    ########################## Download ##########################
    if download:
        file_url = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
        
        # make data raw dir
        current_dir = os.path.dirname(__file__)
        target_dir = os.path.join(current_dir, 'data/raw')
        os.mkdir(target_dir) # create data/raw if doesn't exist
        
        # make data interim and processed dirs
        additional_dir = os.path.join(current_dir, 'data/interim')
        additional_dir_2 = os.path.join(current_dir, 'data/processed')
        os.mkdir(additional_dir)
        os.mkdir(additional_dir_2)
        
        for i in range(len(file_url)):
            downloaded_data = requests.get(file_url[i])
            with open(files_paths[i], 'wb') as f:
                f.write(downloaded_data.content)
    
    ########################## Read MNIST data from zip and write as numpy arrays into dictionnary ##########################
    for path, key in zip(files_paths[:2], files_desc[:2]):
        with gzip.open(path, 'rb') as f:
            data[key] = (np.frombuffer(
                    f.read(), np.uint8, offset=16
                ).reshape(-1, 28 * 28))
            
    for path, key in zip(files_paths[2:], files_desc[2:]):
        with gzip.open(path, 'rb') as f:
            data[key] = np.frombuffer(f.read(), np.uint8, offset=8)


    ########################## Write numpy mnist into file ##########################
    
    # with open('./data/interim/mnist_numpy', 'wb') as f:
    #     pickle.dump(data, f)
        

    ########################## Normalize ##########################

    data[files_desc[0]] = normalize(data[files_desc[0]], black_and_white=True, center=False) # normalize train images
    data[files_desc[1]] = normalize(data[files_desc[1]], black_and_white=True, center=False) # normalize test images
    data[files_desc[2]] = one_hot(data[files_desc[2]]) # one hot train labels
    data[files_desc[3]] = one_hot(data[files_desc[3]]) # one hot test labels
    
    
    ########################## Select train data size ##########################
    if num_train != 60_000:
        indices = np.random.permutation(60_000)
        data[files_desc[0]] = data[files_desc[0]][indices] # shuffle images
        data[files_desc[2]] = data[files_desc[2]][indices] # shuffle labels with same permutation
        
        data[files_desc[0]] = data[files_desc[0]][:num_train] # select num_train elements
        data[files_desc[2]] = data[files_desc[2]][:num_train] # select corresponding num_train elements

    
    ########################## Write dictionary into file ##########################
    with open('./data/processed/mnist_numpy', 'wb') as f:
        pickle.dump(data, f)
        
if __name__=='__main__':
    main(download=True, num_train=60000)