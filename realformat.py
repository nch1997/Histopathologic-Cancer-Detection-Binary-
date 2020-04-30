# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:04:48 2020

@author: Ng Chin Hooi

This section is only for preparing the data for CNN
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#open csv file
dataset = pd.read_csv('train_labels.csv')

#create directory
base_dir = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/train/'
path = [base_dir + str(dataset.iloc[i, 0]) for i in range(len(dataset['label']))]
path_df = pd.DataFrame(path) #convert to dataframe

#join dataframes
data = path_df.join(dataset)

labels = dataset.iloc[:,1]
name = dataset.iloc[:,0]
        
#count unique values in pandas series
labels_count = labels.value_counts()

#plotting bar chart
labels_name = ['No Cancer ( 0 )', ' Cancer ( 1 )']
fig, axs = plt.subplots()
rect1 = axs.bar(labels_name, labels_count)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.title('Number of counts of Histopathologic Images')
plt.show()


#plotting pie chart to see ratio
fig1, ax1 = plt.subplots()
explode = [0.1, 0]
ax1.pie(labels_count, explode = explode, labels=labels_name, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Ratio of Cancer Histopathologic Images')
plt.show()
#%%
#spliting data into 85(train and valid) 15(test) then splitting the 85 into 75 25 for train and valid

#stratify means split the data in proportion based on value given to stratify 
#This stratify parameter makes a split so that the proportion of values 
#in the sample produced will be the same as the proportion of values provided 
#to parameter stratify

#in stratify if y got 59% 0 and 41% 1, the train_test_split will try to split into two sections of 59-41
#For example, if variable y is a binary categorical variable with values 0 and 1 and 
#there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

y = data['label']
df_train_val, df_test = train_test_split(data, test_size = 0.15, 
                                         random_state = 101, stratify = y)

y2 = df_train_val['label']
df_train, df_val = train_test_split(df_train_val, test_size = 0.25, 
                                    random_state = 101, stratify = y2)

train_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/training_model'
valid_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/validation_model'
test_path = 'D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test_model'

# Set the id as the index in df_data so easier manipulate later 
#so can use id as df.loc[id,label]
data.set_index('id', inplace=True)
data.head()

data.rename(columns = {0:'path'}, inplace = True) 

#creating 0 and 1 folders in train_path and valid_path
for fold in [train_path, valid_path, test_path]:
    for subf in ["0", "1"]:
        os.makedirs(os.path.join(fold, subf))

#this 3 loops copy and paste images from one folder to another
# os.path.join combines strings into directory
#shutil.copy copy and paste file from src to dst
for image in df_train['id'].values:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    label = str(data.loc[image,'label']) # get the label for a certain image
    src = os.path.join(base_dir, fname) #join directory base_dir + fname
    dst = os.path.join(train_path, label, fname)
    shutil.copyfile(src, dst)

for image in df_val['id'].values:
    fname = image + '.tif'
    label = str(data.loc[image,'label']) # get the label for a certain image
    src = os.path.join(base_dir, fname)
    dst = os.path.join(valid_path, label, fname)
    shutil.copyfile(src, dst)       

for image in df_test['id'].values:
    fname = image + '.tif'
    label = str(data.loc[image,'label']) # get the label for a certain image
    src = os.path.join(base_dir, fname)
    dst = os.path.join(test_path, label, fname)
    shutil.copyfile(src, dst)    
    
    
sum_train_0 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/training_model/0')))
sum_train_1 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/training_model/1')))
sum_valid_0 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/validation_model/0')))
sum_valid_1 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/validation_model/1')))
sum_test_0 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test_model/0')))
sum_test_1 = (len(os.listdir('D:/Projects/Histopathologic Cancer Detection (CNN)/histopathologic-cancer-detection/test_model/1')))

total_images = sum_train_0 + sum_train_1 + sum_valid_0 + sum_valid_1 + sum_test_0 + sum_test_1
print(total_images)



