#!/usr/bin/env python3
# Behavioral Cloning - Johannes Kadak
# 30 Mar 2017, after a lot of work figuring out that training images need BGR2RGB conversion

#
# Packages
# 
import csv
import cv2
import numpy as np

from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import sys


# 
# Version checks for running in Floyd/etc
# 

print("\n\x1b[32mVersions: \x1b[0m")
print("Keras:", __import__("keras").__version__)
print("TensorFlow:", __import__("tensorflow").__version__)
print("OpenCV:", cv2.__version__)
print("Scikit Learn:", __import__("sklearn").__version__)
print("Numpy:", np.__version__)
print("\n")

# 
# Optional positional arguments
# To use them, simply run: python3 model.py data_dir out_dir
# data_dir contains the driving data (./driving_log.csv, + ./IMG/*)
# out_dir will contain the model (model.h5) and the TensorBoard logs
# 

data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "./"

# 
# State and hyperparameters
# The idea here is to read in all the CSV file lines and read the images 
# batch by batch.
# 

# lines :: [[center, left, right, steering...]], contains all the CSV lines
lines = None

# offset :: float32, contains the offset applied to left/right images when training
OFFSET = 0.25
# batch_size :: int, the size of a batch passed to Keras
BATCH_SIZE = 256
# epochs :: int, how many epochs to run the model
EPOCHS = 7

# process_image :: ndarray(160, 320, 3) -> ndarray(64, 64, 3)
# preprocesses each training image by changing the color space, cropping, resizing 
# and normalizing.
def process_image(image):
    image = image[50:image.shape[0]-25, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = image / 255.0 - 0.5
    return image

# load_image :: [center, left, right...] -> ndarray, float32
# loads a training image according to the CSV line and returns the image
# and its corresponding steering angle.
# the script will randomly take either the left, right or center image
# and apply angle correction to it. it will also randomly flip the image and 
# steering angle.
def load_image(line):
    image_to_take = np.random.choice(3)
    
    measurement = float(line[3])

    # Center
    if image_to_take == 0:
        path = data_dir + "/IMG/" + line[0].split("/")[-1]
    # Left
    elif image_to_take == 1:
        path = data_dir + "/IMG/" + line[1].split("/")[-1]
        measurement += OFFSET
    # Right
    elif image_to_take == 2:
        path = data_dir + "/IMG/" + line[2].split("/")[-1]
        measurement -= OFFSET

    image = cv2.imread(path)
    image = process_image(image)

    # Flip image arbitrarily
    if np.random.uniform() > 0.5:
        image = cv2.flip(image, 1)
        measurement = -1.0 * measurement

    return image, measurement


# 
# Entry point
# 

# Read in the entire list of lines to the array lines.
with open(data_dir + "/driving_log.csv") as csvf:
    reader = csv.reader(csvf)
    next(reader) # Skip header row
    lines = [line for line in reader]

# gen_image_data :: [ [center, left, right, ...] ] -> [image, measurement] x 256
# infinitely generates batches of steering data according to the passed in list of 
# lines to use.
def gen_image_data(lines):
    while True:
        for offset in range(0, len(lines), BATCH_SIZE):
            batch_lines = lines[offset:offset+BATCH_SIZE]
            images = []
            measurements = []
            
            for line in batch_lines:
                image, measurement = load_image(line)
                images.append(image)
                measurements.append(measurement)
            
            X_train = np.array(images)
            Y_train = np.array(measurements)

            yield shuffle(X_train, Y_train)

# 
# Model definition
# 

# The model is a simple addition of dropdown and some parameter mods from LeNet.
# It adds a few layers of dropout and changes around the convolution layers.

model = Sequential()

model.add(Convolution2D(20, (5, 5), padding="same", activation="relu", input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(70, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(120, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(84, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1))

#
# Split the dataset into training and validation data.
# 

lines_train, lines_valid = train_test_split(lines, test_size=0.2)

# Set up generators for training and validation data.
gen_train = gen_image_data(lines_train)
gen_valid = gen_image_data(lines_valid)

# Set up callbacks for the model, to record data into checkpoints and TensorBoard.
checkpoint = ModelCheckpoint("./model-{epoch}.h5")

# note: write_images=True will not work with a generator fit
tb = TensorBoard(log_dir="tb_logs/", histogram_freq=1, write_graph=True, write_images=True)

# Compile the model using the Adam optimizer
model.compile(loss="mse", optimizer="adam", metrics=["acc"])

# Actual training happens here
model.fit_generator(gen_train, 
    steps_per_epoch=len(lines_train) / BATCH_SIZE, 
    validation_data=gen_valid, 
    validation_steps=len(lines_valid) / BATCH_SIZE, 
    epochs=EPOCHS,
    callbacks=[checkpoint, tb])

# Save the final model to model.h5
model.save(out_dir + "/model.h5")
