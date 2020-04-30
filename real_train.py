# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:42:44 2020

@author: User
"""

import os
import shutil
print(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection'))

import tensorflow as tf
import numpy as np
print(tf.__version__)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

train_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/training_model'
valid_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/validation_model'
test_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test_model'


IMAGE_SIZE = 96 #if check properties all image is (96,96,3)
#num_train_samples = len(df_train)
#num_val_samples = len(df_val)
train_batch_size = 32 #normal practice is 32 batch
val_batch_size = 32


image_gen_train = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

image_gen_valid = ImageDataGenerator(rescale = 1./255)

train_gen = image_gen_train.flow_from_directory(train_path,
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = train_batch_size,
                                                class_mode = 'binary')

val_gen = image_gen_valid.flow_from_directory(valid_path,
                                              target_size = (IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size = val_batch_size,
                                              class_mode = 'binary')

test_gen = image_gen_valid.flow_from_directory(test_path,
                                              target_size = (IMAGE_SIZE,IMAGE_SIZE),
                                              batch_size = 1,
                                              class_mode = 'binary')

#%%
#parameters to CNN
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

#dropout stops nodes from learning at random rate to ensure they don rely on 
#only 1 node
dropout_conv = 0.3
dropout_dense = 0.5

#Initializing the CNN
model = Sequential()

#1st layer
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

#2nd layer
model.add(Conv2D(second_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#3rd layer
model.add(Conv2D(third_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#flatten 
model.add(Flatten())

#full connection
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(dropout_dense))
#althought binary is chosen the sigmoid turn the asnwer tovalues between 0 to 1
#the model then use threshold to determine 0 or 1
#e.g <0.5 = 0 >0.5 = 1
model.add(Dense(1, activation = "sigmoid"))

#for alot classes
#softmax gives probability of each belong to which class
#for example with 3 classes
#given image
#ans : [0.6,0.2,0.2] mean first class is answer
#model.add(Dense(num_classes, activation = "softmax"))

# Compile the model
model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics=["accuracy"])
#for more classes
#model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
#training the model
epochs = 100

#stop the training if no improvements
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
#save model every epoch
checkpoint = ModelCheckpoint('checkpoint_model_ori.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit_generator(train_gen,
                              steps_per_epoch=int(np.ceil(140265 / float(train_batch_size))),
                              epochs=epochs,
                              validation_data=val_gen,
                              validation_steps=int(np.ceil(46756 / float(val_batch_size))),
                              callbacks = [checkpoint])
#callbacks = [earlystopper, checkpoint])

#%%
# only use if run until all epoch finish (cannot use with early stopping)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(val_loss))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('firstmodel_ori.h5')

#%%
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

prediction = model.predict(test_gen, steps = 33004, verbose = 1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, prediction)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras