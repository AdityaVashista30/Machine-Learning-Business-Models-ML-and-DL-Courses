# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:54:36 2020

@author: Hp
"""

import tensorflow
from keras_preprocessing.image import ImageDataGenerator

#PART 1: IMAGE PREPROCESSING

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""### Generating images for the Test set"""

test_datagen = ImageDataGenerator(rescale = 1./255)

"""### Creating the Training set"""

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
"""### Creating the Test set"""

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#PART 2: Creating CNN Classifiaction Model
cnn=tensorflow.keras.models.Sequential()

#ADDING FIRST LAYER
#STEP 1: convolution
cnn.add(tensorflow.keras.layers.Conv2D(filters=32,kernel_size=3,input_shape=[64, 64, 3],padding="same", activation="relu"))
#STEP 1: convolution
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#ADDING 2nd LAYER
cnn.add(tensorflow.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#STEP 3: FLATTENING (LAYER FINAL)
cnn.add(tensorflow.keras.layers.Flatten())

#STEP 4: FULL CONNECTION
cnn.add(tensorflow.keras.layers.Dense(units=128, activation='relu'))

"""### Step 5 - Output Layer"""
cnn.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

### Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#FITTING AND TESTING
cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)
