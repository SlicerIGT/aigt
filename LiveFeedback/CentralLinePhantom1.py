import os
import keras
import sys
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.utils import Sequence

import cv2
import matplotlib.pyplot as plt
from pyIGTLink import pyIGTLink

# Parameters

image_size = 128

# Check command line arguments

if len(sys.argv) < 2:
	print("Usage: {} WEIGHTS_FILE".format(sys.argv[0]))
	sys.exit()

weights_file_name = sys.argv[1]

print("Loading weights from: {}".format(weights_file_name))

# Building the model. Should be the same as the weights to be loaded.

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(image_size, image_size, 1)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(.4))
model.add(Dense(2, activation='sigmoid'))

model.load_weights(weights_file_name)

print("Server starting...")
client = pyIGTLink.PyIGTLinkClient(host="127.0.0.1")
client.start()
print("Server running...")

try:
    image_squeezed = np.zeros([image_size, image_size]).astype(np.uint8)
    image_padded = np.zeros([1, image_size, image_size, 1]).astype(np.uint8)
    image_input = np.zeros([1, image_size, image_size, 1]).astype(np.uint8)
    
    cv2.circle(image_input[0,:,:,0], (10,10), 15, 255, 5)
    cv2.imshow("image", image_input[0,:,:,0])
    cv2.waitKey(10)
    
    while True:
      messages = client.get_latest_messages()
      if len(messages) > 0:
        for message in messages:
          if message._type == "IMAGE":
            image = message._image
            image = np.flip(image, 1)
            image_squeezed = np.squeeze(image)
            image_padded[0,:,:,0] = cv2.resize(image_squeezed, (image_size, image_size)).astype(np.uint8)
            image_input = image_padded / 255.0
            prediction = model.predict(image_input).tolist()
            
            print("Predicted center line: " + str(prediction[0]))
            cx = int(image_size*prediction[0][0])
            cy = int(image_size*prediction[0][1])
            cv2.circle(image_input[0,:,:,0], (cx, cy), 2, 255, thickness=1)
            cv2.imshow("image", image_input[0,:,:,0])
            cv2.waitKey(10)
            
            client.send_message(pyIGTLink.StringMessage(str(prediction[0]), device_name=message._device_name+"Predicted"))
      time.sleep(0.1)
except KeyboardInterrupt:
    pass

client.stop()


