# Create and test or load and test the MLP model 
# Usage: python build_mlp_recognizer.py


from config import config

from mnist import MNIST

from sklearn.model_selection import train_test_split
from tensorflow.keras import utils

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint

import os


def preprocess_data(X, y, input_shape, test_size):
  # Reshape letter iamge data into 124800 rows of 28x28 images
  X = X.reshape(config.NUM_DATA_ENTRIES, 28, 28)
  y = y.reshape(config.NUM_DATA_ENTRIES, 1)

  # Recentre to starting index of 1
  y = y - 1

  # Split training data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

  # Reshape 
  X_train.reshape(X_train.shape[0],input_shape[0] ,input_shape[1], input_shape[2])
  X_test.reshape(X_test.shape[0],input_shape[0] ,input_shape[1], input_shape[2])
  
  # Format datatype and scale
  X_train = X_train.astype("float32")/255
  X_test = X_test.astype("float32")/255

  # Encode labels
  y_train = utils.to_categorical(y_train, config.NUM_CLASSES)
  y_test = utils.to_categorical(y_test, config.NUM_CLASSES)

  return (X_train, X_test, y_train, y_test)

def create_MLP_scheme(num_classes, input_shape):
  model = Sequential()
  model.add(Flatten(input_shape = input_shape))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(26, activation='softmax'))

  # Compile model
  model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

  # View model summary
  model.summary()

  return model


def train(model, X_train, y_train, batch_size, epochs):
  checkpointer = ModelCheckpoint(filepath = config.BEST_MLP_MODEL_PATH, verbose = 1, save_best_only=True)
  hist = model.fit(
    X_train, 
    y_train, 
    batch_size = batch_size, 
    epochs = epochs, 
    validation_split = config.MLP_TEST_SIZE,
    callbacks = [checkpointer], 
    verbose = 1,
    shuffle = True)


def main():
  # Load data
  emnist_data = MNIST(path = config.MNIST_DATA_PATH, return_type = "numpy")
  emnist_data.select_emnist("letters")
  X, y = emnist_data.load_training()

  # Preprocess data
  X_train, X_test, y_train, y_test = preprocess_data(X, y, input_shape = (config.IMG_COLS, config.IMG_ROWS, 1), test_size = config.MLP_TEST_SIZE)

  # Create model Scheme
  model = create_MLP_scheme(num_classes = config.NUM_CLASSES, input_shape = (config.IMG_COLS, config.IMG_ROWS, 1)) 

  # Classification accuracy on test set before training
  score = model.evaluate(X_test, y_test, verbose = 0)
  accuracy = 100*score[1]
  print("MLP Test Accuracy Pretraining = {}".format(accuracy))

  # Train a model if one does not exist
  if not os.path.exists(config.MLP_MODEL_PATH):
    train(model, X_train, y_train, batch_size = config.MLP_BATCH_SIZE, epochs = config.MLP_EPOCHS)

    # Load the Model with the Best Classification Accuracy on the Validation Set
    model.load_weights(config.BEST_MLP_MODEL_PATH)

    # Save the best model
    model.save(config.MLP_MODEL_PATH)
  else:
    model = load_model(config.MLP_MODEL_PATH)
  
  
  # Score the model 
  score = model.evaluate(X_test, y_test, verbose = 0)
  accuracy = 100*score[1]
  print("Final MLP Test accuracy: %.4f%%" % accuracy)



  
  

  






if __name__ == '__main__':
  main()