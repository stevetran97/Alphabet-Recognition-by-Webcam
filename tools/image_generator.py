# Image generator
# Generates new 28x28 letter images from the existing mnist letter images to provide more data to train the model.
# List of augmentations used: 
# 1. Rotation 
# 2. Width shift
# 3. Height shift
# 4. Zoom
# 5. Shear

# ----------------------------------------------------------------
# Parameters
rotation_range = 10        # Degrees of rotation 
width_shift_range = 0.2    # Shift image by width percentage
height_shift_range = 0.2   # Shift image by height percentage
shear_range = 0.2          # ???
zoom_range = 0.2           # 
fill_mode = 'nearest'      # Value of pixel to be used to fill the gaps when the image gets rotated (Ideally should be filled with emnishbackground (black))
horizontal_flip = False     # Should not recognize backwards letters (c, e, etc.)



# ----------------------------------------------------------------


from tensorflow.keras import ImageDataGenerator

def main():
  # Instantiate generator
  generator_train_img = ImageDataGenerator(
    rotation_range = rotation_range,        
    width_shift_range = width_shift_range,   
    height_shift_range = height_shift_range,   
    shear_range = shear_range,            
    zoom_range = zoom_range,
    horizontal_flip = horizontal_flip,
    fill_mode = fill_mode
  )



if __name__ == __main__:
  main()