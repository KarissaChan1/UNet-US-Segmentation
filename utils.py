import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
import os
from PIL import Image, ImageFilter
import cv2
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import random


def preprocess_images(image, annotation,dim,resize_only=False):
    # Convert to grayscale
    image_gray = image.convert('L')
    ann_gray = annotation.convert('L')

    if resize_only==False:

        # Apply blur
        image_blurred = image_gray.filter(ImageFilter.BLUR)
        
        # Augmentation
        aug_img = iaa.Sequential([
          iaa.GaussianBlur(sigma=(0, 3.0)),
          iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
          iaa.Multiply((0.8, 1.2))
        ])
        

        rotation_random = random.uniform(-45,45)
        rotation_augmentation = iaa.Affine(rotate=(rotation_random))
        
        #Crop/resize
        image_augmented = aug_img.augment_image(np.array(image_blurred))
        image_augmented = rotation_augmentation(images=np.array(image_augmented,dtype=np.uint8))
        ann_augmented = rotation_augmentation(images=np.array(ann_gray,dtype=np.uint8))

      
        image_augmented = tf.convert_to_tensor(image_augmented)
        image_augmented = tf.expand_dims(image_augmented,-1)
        image_augmented = tf.image.resize_with_crop_or_pad(image_augmented,*dim)

        ann_augmented = tf.convert_to_tensor(ann_augmented)
        ann_augmented = tf.expand_dims(ann_augmented,-1)
        ann_augmented = tf.image.resize_with_crop_or_pad(ann_augmented,*dim)

        ann_augmented = tf.cast(ann_augmented,tf.float32)/255.0
        ann_augmented=tf.cast(ann_augmented,tf.int32)

    else:

        #Crop/resize
        image_augmented = np.array(image_gray)
        ann_augmented = np.array(ann_gray)

        image_augmented = tf.convert_to_tensor(image_augmented)
        image_augmented = tf.expand_dims(image_augmented,-1)
        image_augmented = tf.image.resize_with_crop_or_pad(image_augmented,*dim)

        ann_augmented = tf.convert_to_tensor(ann_augmented)
        ann_augmented = tf.expand_dims(ann_augmented,-1)
        ann_augmented = tf.image.resize_with_crop_or_pad(ann_augmented,*dim)

        ann_augmented = tf.cast(ann_augmented,tf.float32)/255.0
        ann_augmented=tf.cast(ann_augmented,tf.int32)

    return image_augmented,ann_augmented
    
def scale_resize_image(image, label,resize=(540,800)):
    image = tf.convert_to_tensor(image) # equivalent to dividing image pixels by 255
    image = tf.expand_dims(image,-1)
  
    label = tf.convert_to_tensor(label)
    label = tf.expand_dims(label,-1)
    
    if resize!=None:
        image = tf.image.resize_with_crop_or_pad(image,*resize)
        label = tf.image.resize_with_crop_or_pad(label,*resize)
        
    label = tf.cast(label, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    
    return image, label


def augment(im,label,resize=(540,800)):
    im,label = scale_resize_image(im,label,resize)
#     im,label = preprocess_images(im,label,dim=(640,640),resize_only=False)
    return im, label

def dice_coeff(y_true, y_pred):
    smooth = 1.
    
#     y_pred_bin = tf.cast(y_pred > 0.5, tf.uint8)
#     # Flatten
    y_true = tf.cast(y_true, dtype=tf.float32)  # Convert labels to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32)
#     y_pred = tf.round(y_pred)
#     y_true = tf.round(y_true)
    
#     y_true = tf.cast(y_true, dtype=tf.float32)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def get_test_images(path,dim,batch_size):
    X = np.zeros((batch_size,*dim,1))
    files = os.listdir(path)
    
    for i in range(batch_size):
        image = Image.open(path+files[i])
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image,-1)
        image = tf.image.resize_with_crop_or_pad(image,*dim)

        X[i,]=image
    
    return X

def overlay_masks(img,label):
    label = label.astype(np.uint8)
    heatmap_img = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    roi_img = cv2.bitwise_and(overlay_img, overlay_img, mask = label)
    roi_img[label[:] == 0,...] = img[label[:] == 0,...]
    return roi_img
    