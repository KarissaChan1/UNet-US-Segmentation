import numpy as np
import keras
from utils import augment
from PIL import Image
import tensorflow
import os

class CustomDataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size=32, dim=(32,32,32), n_channels=1,
                 shuffle=True, augmentation=True,resize=None):
        'Initialization'
        self.dim = dim # dimensions of images (make sure they are all the same dimension)
        self.batch_size = batch_size # choose batch size
        self.n_channels = n_channels # = 1 for grayscale
        self.shuffle = shuffle
        self.augmentation = augmentation # whether or not you want to perform augmentation
        self.resize=resize
        self.path = path
        self.img_folder = self.path + 'Image/'
        self.mask_folder = self.path + 'Annotation/'
        self.list_IDs = os.listdir(self.img_folder) # make sure the names of the corresponding files in the images and annotations folders are the same
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            ID_split,_ = os.path.splitext(ID) 

            img = Image.open(self.img_folder + ID_split + '.png') # pls double check this function is correct for importing .png files
            label = Image.open(self.mask_folder + ID_split + '_Annotation.png')

            
            if self.augmentation==True:
                augmented_img, augmented_label = augment(img, label,self.dim)
                X[i,] = augmented_img
                y[i,] = augmented_label
#                 print(np.unique(augmented_label))
            else:
#                 img,label = preprocess_images(img,label,self.dim,resize_only=True)
                X[i,]=img
                y[i,]=label
                 

        return X, y