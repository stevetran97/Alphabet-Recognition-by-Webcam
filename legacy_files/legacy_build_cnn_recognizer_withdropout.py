# Legacy CNN Builder 1
# Create and test or load and test the CNN model 
# Usage: python build_cnn_recognizer.py

# Original CNN builder file.
# Model uses Dropouts instead of Batch Normalization
# Does not use augmented sample images

# ----------------------------------------------------------------


from config import config
from mnist import MNIST
from sklearn.model_selection import train_test_split

import keras
from tensorflow.keras import utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras import backend as K
from mnist import MNIST
from sklearn.model_selection import train_test_split

import os

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
  
  # Convert arrays to float32 type and scale between [0,1]
  x_train = x_train.astype("float32")/255
  x_test = x_test.astype("float32")/255

  # Encode labels
  y_train = utils.to_categorical(y_train, config.NUM_CLASSES)
  y_test = utils.to_categorical(y_test, config.NUM_CLASSES)

  return (x_train, x_test, y_train, y_test)


def create_CNN_scheme(num_classes, input_shape):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

  # View model summary
  model.summary()
  return model


def train_model(model, x_train, y_train, batch_size, epochs, x_test, y_test):
  model.fit(x_train, 
          y_train, 
          batch_size = batch_size, 
          epochs = epochs,
          verbose = 1,
          validation_data = (x_test, y_test))

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










