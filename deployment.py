# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:01:32 2020

@author: User
"""

from tensorflow.keras.models import load_model 
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#to show image
#from PIL import Image
#img = Image.open(r'test_model\1\00a2a1175108c1c63970e01b71e664cccc10e5ec.tif')
#img.show()

model = load_model('checkpoint_model.h5')
model.summary()


from tensorflow.keras.preprocessing import image
test_image = image.load_img(r'test\85dc6972146c8cb956afbbf34f3876e738b555b1.tif', target_size = (96, 96))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image /= 255. #REMEBMER TO SCALE DATA
result = model.predict(test_image)