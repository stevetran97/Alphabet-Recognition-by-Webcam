# Shared Config File for Application
# Contains configurable variables 
import os
import numpy as np



# ----------------------------------------------------------------
# Training Configuration
# Data path
MNIST_DATA_PATH = "data\\"
# Number of data entries/images in the database
NUM_DATA_ENTRIES = 124800

# path to which model is built
MLP_MODEL_PATH = os.path.sep.join(['outputs\\models', 'eminst_mlp_model.h5'])
CNN_MODEL_PATH = os.path.sep.join(['outputs\\models', 'emnist_cnn_model.h5'])
BEST_MLP_MODEL_PATH = os.path.sep.join(['outputs\\models\\', 'emnist.model.best.hdf5'])

# Shared model Variables
NUM_CLASSES = 26  # Total number of possible classes (26 alphabet letters)
IMG_ROWS, IMG_COLS = 28,28  # Input image dimensions
# To decode results of the model (starting index: 1)
LETTERHASH = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}


# CNN model configuration
CNN_TEST_SIZE = 0.25  # Fractional test-train data split
CNN_BATCH_SIZE = 128  # Number of images in each training batch
CNN_EPOCHS = 500

# MLP model configuration
MLP_TEST_SIZE = 0.25
MLP_BATCH_SIZE = 128  # Number of images in each training batch
MLP_EPOCHS = 75  # Number of Epochs to study each letter



# ----------------------------------------------------------------
# Recognizer configuration
# Upper and lower RGB object value limits to detect the object on the screen
# The ACTUAL HSV value must be converted prior to use in OpenCV functions
# Input to HSV function is denoted by (Real Hue Value / 2, (Real % Sat * 255), (Real % Value * 255))

# Orange Setting
detectHSVLower = np.array([10, 130, 130]) # (Real Hue = 20, Real Sat = 130/255 * 100, Real V = 130/255*100)
detectHSVUpper = np.array([25, 255, 255]) # (Real Hue = 50, Real Sat = 100%, Real V = 100%)

# Green Setting
# detectHSVLower = np.array([40, 60, 60])
# detectHSVUpper = np.array([85, 255, 255])

# Blue setting
# detectHSVLower = np.array([40, 60, 60])
# detectHSVUpper = np.array([85, 255, 255])


# Letter image blackboard size
# Warning! Drawn letter must be drawn within these space constraints!
LETTER_BLACKBOARD = np.zeros((200, 200, 3), dtype = np.uint8)

# Dilation and Erosion Kernel
# Used to eliminate colour noise and/or strengthen the object colours
KERNEL = np.ones((5,5), np.uint8) 

# Minimum area for dsicovered blackboard contours for which the algorithm will to try to predict the blackboard
MIN_LETTER_CONTOUR_AREA = 1000


# ----------------------------------------------------------------
# CNN Training Data Augmentation Config
# Only used in legacy data augmentation build
AUG_ROTATION_RANGE = 10        # Degrees of rotation 
AUG_WIDTH_SHIFT_RANGE = 0.05    # Shift image by width percentage
AUG_HEIGHT_SHIFT_RANGE = 0.05   # Shift image by height percentage
# AUG_SHEAR_RANGE = 0.2          # ???
AUG_ZOOM_RANGE = 0.05           # 
AUG_FILL_MODE = 'nearest'      # Value of pixel to be used to fill the gaps when the image gets rotated (Ideally should be filled with emnishbackground (black))
AUG_HORIZONTAL_FLIP = False     # Should not recognize backwards letters (c, e, etc.)





