import csv
import cv2
import numpy as np

from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import Sequential
from keras.callbacks import ProgbarLogger, ModelCheckpoint 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "./"

lines = None
OFFSET = 0.25
BATCH_SIZE = 256

images = []
measurements = []

def process_image(image):
    image = image[50:image.shape[0]-25, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = image / 255.0 - 0.5
    return image

# Load an image.
# Randomly decides to take images from the left/right/center camera and
# randomly flips them.
# Keeps data count constant while providing augmentation

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
# Entry point: load data and images.
# 

with open(data_dir + "/driving_log.csv") as csvf:
    reader = csv.reader(csvf)
    next(reader) # Skip header row
    lines = [line for line in reader]

for line in lines:
    image, measurement = load_image(line)
    images.append(image)
    measurements.append(measurement)

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

# Limit the size of training data to finish training on my laptop in meaningful time
lines_train, lines_valid = train_test_split(lines, train_size=4096, test_size=1024)

gen_train = gen_image_data(lines_train)
gen_valid = gen_image_data(lines_valid)

# callbacks
checkpoint = ModelCheckpoint("./model-{epoch}.h5")

model.compile(loss="mse", optimizer="adam", metrics=["acc"])
model.fit_generator(gen_train, 
    steps_per_epoch=len(lines_train) / BATCH_SIZE, 
    validation_data=gen_valid, 
    validation_steps=len(lines_valid) / BATCH_SIZE, 
    epochs=7)

model.save(out_dir + "/model.h5")
