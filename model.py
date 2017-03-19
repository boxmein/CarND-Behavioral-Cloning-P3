 #!/usr/bin/env python3
# encoding: utf8
# CarND: Behavioral Cloning
# - Johannes Kadak 2017

# https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.hktkdj33j
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.hid7wp2m6
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.hid7wp2m6



# Data reading
import csv
from scipy.misc import imread
import itertools

# Model
import numpy as np
from keras.layers import *
from keras.models import Sequential

# Data reader

def load_image(filename):
    return imread(filename,
                      flatten=False,
                      mode='RGB')

def load_csv(data_dir='../data/'):
    with open(data_dir + "driving_log.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        # Skip a line
        next(csv_reader, None)
        for center_img, left_img, right_img, steer_angle, throttle, brake, speed in csv_reader:
            yield load_image(data_dir + center_img), float(steer_angle)

#    Load CSV
#    Yield images:
#     (Center, left, right):
#      Original
#      Flipped horizontally
#      Random coloring: cv2.cvtColor will help here

##
# Hyperparameters
OPTIMIZER = 'adam'
LOSS = 'mse'
EPOCHS = 10
VALIDATION_SPLIT=0.2
##

def behavioral_cloning():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    return model

model = behavioral_cloning()

# For starters, consume data directly into memory
data = list(itertools.islice(
        load_csv("./data/"),
        1000))

X_train = np.array(list(map(lambda x: x[0], data)))
y_train = np.array(list(map(lambda x: x[1], data)))

print("X Training:", X_train.shape)
print("Y Training:", y_train.shape)

model.fit(X_train, y_train,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    epochs=EPOCHS)
model.save("model.h5")
