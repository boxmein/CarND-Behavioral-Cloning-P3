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

# Model
import numpy as np

import cv2

import keras
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import Sequential

import tensorflow as tf

N_TRAIN = 200

##
# Hyperparameters

OPTIMIZER = 'adam'
LOSS = 'mse'
EPOCHS = 5
VALIDATION_SPLIT=0.1

ORIG_ROWS = 160
ORIG_COLS = 320
ORIG_CH = 3

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

def reverse_image(image):
    pass

def color_jitter(image):
    pass

def crop_image(image):
    return image[50:150, :, :]

def resize_image(image):
    image = tf.image.resize_images(x, (IMG_ROWS, IMG_COLS))
    return image

def preprocess_image(image):
    return resize_image(
        crop_image(
            image
        )
    )

def data_augmenter(image_gen):
    for image, angle in image_gen:
        yield image, angle

def load_image(filename):
    return imread(filename,
                      flatten=False,
                      mode='RGB')

def load_csv(data_dir='/input/'):
    with open(data_dir + "driving_log.csv") as csv_file:
        xx = False
        csv_reader = csv.reader(csv_file)
        # Skip the headers
        next(csv_reader, None)
        
        csv_reader = itertools.islice(csv_reader, N_TRAIN)
        csv_reader = itertools.cycle(csv_reader)
        
        for center_img, left_img, right_img, steer_angle, throttle, brake, speed in csv_reader:
            image = load_image(data_dir + center_img)
            
            steer_angle = float(steer_angle)
            
            if -0.0001 < steer_angle < 0.0001:
                yield np.resize(image, [1, ORIG_ROWS, ORIG_COLS, ORIG_CH]), np.array([steer_angle])

def behavioral_cloning():
    
    model = Sequential()
    
    # Preprocessing in the model from @mohankarthik

    #model.add(Lambda(lambda x: x[50:150, :, :], name="ImageCropper", input_shape=(ORIG_ROWS, ORIG_COLS, ORIG_CH)))
    #model.add(Lambda(lambda x: tf.image.resize_images(x, (IMG_ROWS, IMG_COLS)), name="ImageResizer"))
    #model.add(Lambda(lambda x: x/127.5 - .5, name="ImageNormalizer"))

    # Color Space layer from @mohankarthik
    model.add(Convolution2D(3, 1, padding='same', name="ColorSpaceConv", input_shape=(ORIG_ROWS, ORIG_COLS, ORIG_CH)))

    ###########
    # Model:
    ###########
    
    # Convolution + ReLU
    model.add(Conv2D(16, 7, strides=2, activation="relu", use_bias=True))
    # Max pooling
    model.add(MaxPooling2D((2,2), (1,1)))

    # Convolution + ReLU
    model.add(Conv2D(16, 5, strides=2, activation="relu", use_bias=True))
    # Max pooling
    model.add(MaxPooling2D((2,2), (1,1)))

    # Convolution + ReLU
    model.add(Conv2D(16, 3, strides=2, activation="relu", use_bias=True))
    # Dropout
    model.add(Dropout(0.5))
    model.add(Flatten())

    # First dense layer
    model.add(Dense(256))
    model.add(Dropout(0.5))

    # Second dense layer
    model.add(Dense(128))
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
        help='Data folder prefix. Add a slash to the end.'
    )
    parser.add_argument(
        'logs_dir',
        type=str,
        default="tb_logs/",
        help='Logs folder prefix. Add a slash to the end.'
    )
    
    args = parser.parse_args()
    
    model = behavioral_cloning()
    
    data = data_augmenter(load_csv(args.data_dir))
    
    tb = keras.callbacks.TensorBoard(log_dir=args.logs_dir, write_graph=True)
    cp = keras.callbacks.ModelCheckpoint("model-epoch-{epoch:02d}.h5")
    
    # Keras 1.2 mods: epochs -> nb_epoch
    model.fit_generator(data,
        steps_per_epoch=N_TRAIN - (VALIDATION_SPLIT * N_TRAIN),
        validation_steps= VALIDATION_SPLIT * N_TRAIN,
        epochs=EPOCHS,
        callbacks=[tb, cp])
    model.save("model.h5")
