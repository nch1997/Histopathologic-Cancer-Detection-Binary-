# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:29:29 2020

@author: User

This section of code is to test the model but not for deployment still requires data to be in specific folders
for testing purpose only
"""


import tensorflow as tf
import numpy as np
print(tf.__version__)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 96 #if check properties all image is (96,96,3)


from tensorflow.keras.models import load_model 
model = load_model('checkpoint_model_ori.h5')
model.summary()

train_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/training_model'
valid_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/validation_model'
test_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test_model'
#test_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test'
datagen = ImageDataGenerator(rescale = 1./255)
test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='binary',
                                        shuffle=False) #tells if data should be shuffled anot default is true

#%%
y_pred_keras = model.predict(test_gen, steps=33004, verbose=1)


# Data cannot be shuffled
from sklearn.metrics import roc_curve, auc, roc_auc_score
#Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. 
#By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.
#to plot AUC need FRP and TPR first 
#https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 

fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#%%
# To do this data must not be shuffled
# Chg line 41
y_pred = (y_pred_keras > 0.5)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(test_gen.classes, y_pred)

acc = accuracy_score(test_gen.classes, y_pred)

#%%
# Data can be shuffled here
# Chg line 41
acc2 = model.evaluate_generator(generator = test_gen, steps = 33004)


