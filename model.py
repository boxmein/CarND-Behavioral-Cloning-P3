#!/usr/bin/env python3
# encoding: utf8
# CarND: Behavioral Cloning
# - Johannes Kadak 2017

# https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.hktkdj33j
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.hid7wp2m6
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.hid7wp2m6



# Data reading
import csv
from scipy.misc import imread, imsave
import itertools
import argparse

# Model
import numpy as np
import pandas as pd

import cv2
import random

import keras
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import Sequential

import tensorflow as tf

##
# Hyperparameters
N_TRAIN = 1600
TRAIN_IN_STEPS=20
BATCH_SIZE = N_TRAIN // TRAIN_IN_STEPS
VALIDATION_STEPS = 4

OPTIMIZER = 'adam'
LOSS = 'mse'
EPOCHS = 10
VALIDATION_SPLIT=0.1

IMG_ROWS = 64
IMG_COLS = 64
IMG_CH = 3
##

print("VERSIONS")
print("Keras: ", keras.__version__)
print("Numpy: ", np.__version__)
print("OpenCV: ", cv2.__version__)
print("CONSTANTS")
print("N_TRAIN = ", N_TRAIN)
print("OPTIMIZER = ", OPTIMIZER)
print("LOSS = ", LOSS)
print("EPOCHS = ", EPOCHS)
print("VALIDATION_SPLIT = ", VALIDATION_SPLIT)
print("IMG_ROWS = ", IMG_ROWS)
print("IMG_COLS = ", IMG_COLS)
print("IMG_CH = ", IMG_CH)


def load_image(filename):
    return cv2.imread(filename)

def load_csv(data_dir='/input/'):
    df = pd.read_csv(data_dir + "driving_log.csv", 
                delimiter=',',
                header=0,
                names=['center','left','right','steering','throttle','brake','speed'])
    segment = df.sample(n=N_TRAIN)
    
    images = segment["center"]
    images = np.array([load_image(data_dir + i) for _, i in images.iteritems()])
    print("Images loaded", images.shape)

    angles = segment.apply(lambda row: float(row['steering']), axis=1).as_matrix()
    print("Angles loaded", angles.shape)

    while 1:
        for i in range(TRAIN_IN_STEPS//2):
            sub_images = images[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            sub_flip = np.empty(shape=sub_images.shape)
            
            for idx, img in enumerate(sub_images):
                sub_flip[idx] = cv2.flip(img, 1)
            
            sub_angles = angles[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            
            mask = (sub_angles < -0.001) | (sub_angles > 0.001)
            
            yield sub_images[mask], sub_angles[mask]
            yield sub_flip[mask], -sub_angles[mask]
            
def behavioral_cloning():
    
    model = Sequential()
    
    model.add(Lambda(lambda x: __import__("tensorflow").image.resize_images(x, (64, 64)), input_shape=(160, 320, 3)))

    # model.add(Lambda(lambda x: x / 255.0 - 0.5, name="ImageNormalizer"))
    
    # Color Space layer from @mohankarthik
    # model.add(Convolution2D(3, 1, padding='same', name="ColorSpaceConv", input_shape=(160, 320, 3)))

    ###########
    # Model:
    ###########
    
    # Convolution + ReLU
    model.add(Conv2D(4, 9, strides=1, activation="relu", use_bias=True, input_shape=(160, 320, 3)))
    model.add(Conv2D(4, 7, strides=1, activation="relu", use_bias=True))
    # Max pooling
    model.add(MaxPooling2D((2,2), (1,1)))

    # Convolution + ReLU
    model.add(Conv2D(8, 7, strides=1, activation="relu", use_bias=True))
    model.add(Conv2D(8, 5, strides=1, activation="relu", use_bias=True))
    # Max pooling
    model.add(MaxPooling2D((2,2), (1,1)))

    # Convolution + ReLU
    model.add(Conv2D(12, 5, strides=1, activation="relu", use_bias=True))
    model.add(Conv2D(12, 3, strides=1, activation="relu", use_bias=True))
    # Max pooling
    model.add(MaxPooling2D((2,2), (1,1)))

    # Convolution + ReLU
    model.add(Conv2D(16, 3, strides=1, activation="relu", use_bias=True))
    model.add(MaxPooling2D((2,2), (1,1)))

    model.add(Dropout(0.5))
    model.add(Flatten())

    # First dense layer
    model.add(Dense(256))
    model.add(Dropout(0.5))

    # Second dense layer
    model.add(Dense(64))
    model.add(Dropout(0.5))
    
    # Third dense layer
    model.add(Dense(1))

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'data_dir',
        type=str,
        default="data/",
        nargs="?",
        help='Data folder prefix. Add a slash to the end.'
    )
    parser.add_argument(
        'logs_dir',
        type=str,
        default="tb_logs/",
        nargs="?",
        help='Logs folder prefix. Add a slash to the end.'
    )

    parser.add_argument(
        'out_dir',
        type=str,
        default="./",
        nargs="?",
        help='Output folder prefix. Add a slash to the end.'
    )
    
    args = parser.parse_args()
    
    model = behavioral_cloning()
    # model = nvidia_model()
    
    data = load_csv(args.data_dir)
    
    tb = keras.callbacks.TensorBoard(log_dir=args.logs_dir, write_graph=True)
    cp = keras.callbacks.ModelCheckpoint("model-epoch-{epoch:02d}.h5")
    
    # Keras 1.2 mods: epochs -> nb_epoch
    model.fit_generator(data,
        steps_per_epoch=TRAIN_IN_STEPS - VALIDATION_STEPS,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS,
        callbacks=[tb, cp])
    model.save(args.out_dir + "model.h5")
