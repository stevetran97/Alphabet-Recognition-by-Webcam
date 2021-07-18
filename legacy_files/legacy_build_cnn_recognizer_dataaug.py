# Create and test or load and test the CNN model 
# Usage: python build_cnn_recognizer.py


# Notes: This build trains the model properly and is capable of incrementally increasing accuracy but tensorflow bugs prevent it from training optimally.

# ----------------------------------------------------------------

import numpy
from config import config
from mnist import MNIST
from sklearn.model_selection import train_test_split

import keras
from tensorflow.keras import utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras import backend as K
from mnist import MNIST
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# DEBUGGING
import numpy as np

def preprocess_data(X, y, input_shape, test_size):
  # Loading letter image data
  # Reshape data into 124800 rows of 28x28 images
  X = X.reshape(config.NUM_DATA_ENTRIES, 28, 28)
  y = y.reshape(config.NUM_DATA_ENTRIES, 1)

  # Recentre labels to start at python 0 indices
  y = y-1

  #Split training-test data
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

  # Data Reshaping for CNN Input
  x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[0], input_shape[2])
  x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[0], input_shape[2])
  
  # ------------------------------------------------------------------
  # Bug: Changing value types appears to make the input incompatible with augmented data generation
  # Suggests that augmenting data needs to know exact Greyscale values in the array.
  # # Convert arrays to float32 type and scale between [0,1]
  # x_train = x_train.astype("float32")/255
  # x_test = x_test.astype("float32")/255
  # ------------------------------------------------------------------

  # Encode labels
  y_train = utils.to_categorical(y_train, config.NUM_CLASSES)
  y_test = utils.to_categorical(y_test, config.NUM_CLASSES)

  return (x_train, x_test, y_train, y_test)


def create_CNN_scheme(num_classes, input_shape):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(BatchNormalization())
  # model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(BatchNormalization())
  # model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

  # View model summary
  model.summary()
  return model


def train_model(model, x_train, y_train, batch_size, epochs, x_test, y_test):
  # Fit model: with generated data
  # Augmented Image Preprocessing prior to model feed
  generator_train_img = ImageDataGenerator(
    rotation_range = config.AUG_ROTATION_RANGE,        
    width_shift_range = config.AUG_WIDTH_SHIFT_RANGE,   
    height_shift_range = config.AUG_HEIGHT_SHIFT_RANGE,   
    shear_range = config.AUG_SHEAR_RANGE,            
    zoom_range = config.AUG_ZOOM_RANGE,
    horizontal_flip = config.AUG_HORIZONTAL_FLIP,
    fill_mode = config.AUG_FILL_MODE,
    validation_split=0.2,
    # Rescaling must be done in the data generation
    rescale = 1./255
  )

  # Generator is based on x_train
  generator_train_img.fit(x_train)

  # model fitting directly takes generated data during training
  # Issue 1: Without steps_per_epoch, epoch 1 trains forever
  # Issue 2: With steps_per_epoch, model.fit hangs at the end of epoch 1.
  # Solution for next iteration: Generate training data set of new but larger fixed size but store as a variable in the function. Pass finite training data to model.fit 
  # Cons -> Limited memory
  # Solution pt2: Generate training data set but draw new images in an output file. Copy and paste generated images into the training folder.
  # Cons -> Need to append in same format with Emnist images. Could lead to unintentional side effects
  model.fit(generator_train_img.flow(x_train, y_train, batch_size = batch_size, subset='training'), 
          epochs = epochs,
          #Steps_per_epoch and validation_steps must equal len(x_train)/batch_size.  Otherwise, tensorflow hangs on the later versions
          #Steps_per_epoch is the number of BATCHES of sample images to use in one epoch. Ie. One epoch = steps_per_epoch*batch_size images used to train it. In total, the number of images used in training in one epoch will be equal to the number of training images.
          #In this case, it seems that 93600//32 randomly generated images will be used to train the model in each epoch.
          #If it is any less, you will run out of data for that epoch (Not sure why that is bad)
          #Statistically, each image will be slightly different from the last. 
          #Potential problems: Does the generator use each image in the training folder AT LEAST once to generate a SINGLE new randomly generated image or does it pick randomly? Is there a chance that one image is used twice in the generation and training process? Or does each training image get used ONCE for EVERY EPOCH?
          #Ideally, the same image should not be used more than once. All images should be used in each Epoch in order to maintain data diversity and balance.
          steps_per_epoch=93600//32,
          validation_steps= 31200//32,
          batch_size = batch_size,
          verbose = 1,
          #validation_data = generator_train_img.flow(x_train, y_train, batch_size = 8, subset='validation')
          validation_data = (x_test, y_test)
          )

  # ------------------------------------------------------------------
  # Alternate Fit model:  with non-generated data
  # model.fit(x_train, 
  #         y_train,
  #         epochs = epochs,
  #         batch_size = batch_size,
  #         verbose = 1,
  #         validation_data = (x_test, y_test)         
  #         )
  # ------------------------------------------------------------------


def main():
  # Load raw data
  emnist_data = MNIST(path = config.MNIST_DATA_PATH, return_type = 'numpy')
  emnist_data.select_emnist('letters')
  X, y = emnist_data.load_training()

  # Preprocess data
  x_train, x_test, y_train, y_test = preprocess_data(X, y, input_shape = (config.IMG_ROWS, config.IMG_COLS, 1), test_size = config.CNN_TEST_SIZE)

  # Instantiate model structure
  model = create_CNN_scheme(input_shape = (config.IMG_ROWS, config.IMG_COLS, 1), num_classes = config.NUM_CLASSES)
  
  # Save new OR load existing model
  if not os.path.exists(config.CNN_MODEL_PATH):
    # Train model
    train_model(model, x_train, y_train, config.CNN_BATCH_SIZE, config.CNN_EPOCHS, x_test, y_test)
    # Save model weights
    model.save(config.CNN_MODEL_PATH)
    print('[INFO] New CNN Saved:', config.CNN_MODEL_PATH)
  else:
    model = load_model(config.CNN_MODEL_PATH)
    print('[INFO] Existing CNN Loaded:', config.CNN_MODEL_PATH)

  # Score the model
  score = model.evaluate(x_test, y_test, verbose = 0)
  print('[INFO] Test Loss:', score[0])
  print('[INFO] Test Accuracy:', score[1])

if __name__ == '__main__':
  main()










