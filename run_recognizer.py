# Recognizer app
# Uses webcam to start detecting writing "object". Traces the object to get a digit. Predicts the digit with MLP and CNN models
# use: python run_recognizer.py
# -----------------------------------------------------------------------
from config import config
from keras.models import load_model
import numpy as np

from collections import deque
import cv2

# Create an empty blackboard and adjacent points array for character drawing
def createEmptyBlackBoard():
  drawn_points = deque(maxlen = 512)
  blackboard = np.zeros((480, 640, 3), dtype = np.uint8)

  return (drawn_points, blackboard)

# Filters frame to expose and extract drawing object contour
def getPotenDrawObjContours(frame):
  # Processed frames for better detection or prediction
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
  # Create a filtered Mask to find the drawing object
  color_mask = cv2.inRange(hsv, config.detectHSVLower, config.detectHSVUpper)  # Remove non-drawing activation colours
  color_mask = cv2.erode(color_mask, config.KERNEL, iterations = 2)  # Filter faint/dim object colours
  color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, config.KERNEL)  # Remove noise around the drawing object
  color_mask = cv2.dilate(color_mask, config.KERNEL, iterations = 1) # Exagerate remaining object colours

  # Get all potential drawing object contours 
  (drawer_contours, _) = cv2.findContours(color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  return drawer_contours

# Find the center coordinates of a contour relative to its parent frame
def findContourCentre(frame, contour):
  ((x, y), radius) = cv2.minEnclosingCircle(contour)
  cv2.circle(frame, (int(x), int(y)), int(radius), (117, 0, 135), 2)

  M = cv2.moments(contour)
  centre = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

  return centre

# Filters blackboard frame to expose and isolate the drawn letter
def getPotenLetterContours(blackboard):  
  blurred_board = cv2.medianBlur(blackboard, 15) # Erode rough surfaces
  blurred_board = cv2.GaussianBlur(blurred_board, (5, 5), 0)  # Erode all surfaces
  blurred_board = cv2.dilate(blurred_board, config.KERNEL, iterations = 2) # Exagerate letter thickness
  threshed_board = cv2.threshold(blurred_board, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Remove grey noise

  # Get all potential letter contours
  (blackboard_contours, _) = cv2.findContours(threshed_board.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  return blackboard_contours

def predictBlackBoardImage(letter_image, mlp_model, cnn_model):
  # Preprocess image for model input
  newImage = cv2.resize(letter_image, (28, 28))
  newImage = np.array(newImage, dtype = np.float32)/255

  # Make predictions
  mlp_prediction = mlp_model.predict(newImage.reshape(1, 28, 28))[0]
  mlp_prediction = np.argmax(mlp_prediction)

  cnn_prediction = cnn_model.predict(newImage.reshape(1 ,28, 28, 1))[0]
  cnn_prediction = np.argmax(cnn_prediction)

  return (mlp_prediction, cnn_prediction)

# Primary application
def main():
  #  Load models
  mlp_model = load_model(config.MLP_MODEL_PATH) 
  cnn_model = load_model(config.CNN_MODEL_PATH)
  print("\n\n\n[INFO] Loaded Models")

  # Define a blackboard to contain the darwn letter
  letter_image = config.LETTER_BLACKBOARD

  # Setup empty deque and blackboard for drawing
  (drawn_points, blackboard) = createEmptyBlackBoard()

  # Instantiate prediction variable values (26 = No current prediction)
  mlp_prediction = 26
  cnn_prediction = 26

  # Access primary web camera 
  camera = cv2.VideoCapture(0)
  print("\n\n\n[INFO] Setup Variables and Camera Loaded")

  # Captures frames from webcam on infinite loop
  while True:
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame

    # Helper finds all potential drawing objects in current frame
    draw_contours = getPotenDrawObjContours(frame)

    draw_centre = None  # Reset draw object centre
    # 1. Drawing state: Continously add points at the centre of the drawing object if it is on screen
    if len(draw_contours) > 0:
      draw_contour = sorted(draw_contours, key=cv2.contourArea, reverse = True)[0]  # Get the first (largest area) contour
      # Get centre coordinates of draw contour relative to the frame
      draw_centre = findContourCentre(frame, draw_contour)
      drawn_points.appendleft(draw_centre)

    # 2. Predict state: Course of action when drawing object/contour disappears from screen and there are more than one point drawn on the blackboard
    elif len(draw_contours) == 0:
      if len(drawn_points) != 0:
        blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)  # Create greyscale of drawing board

        # Helper finds all potential letters on the drawing board
        blackboard_contours = getPotenLetterContours(blackboard_gray)

        if len(blackboard_contours) >= 1:
          # Get largest blackboard letter contour
          contour = sorted(blackboard_contours, key = cv2.contourArea, reverse = True)[0]

          if cv2.contourArea(contour) > config.MIN_LETTER_CONTOUR_AREA:
            # Extract bounding box image of letter
            x, y, w, h = cv2.boundingRect(contour)
            # Model preprocessing: extract letter image as an array, resize to input to model, standardize 
            letter_image = blackboard_gray[y-10:y + h + 10, x - 10:x + w + 10]

            (mlp_prediction, cnn_prediction) = predictBlackBoardImage(letter_image, mlp_model, cnn_model)

          # Empty points deque and blackboard
          (drawn_points, blackboard) = createEmptyBlackBoard()

          
    # Draw on blackboard by connecting all contour centre points found
    for i in range(1, len(drawn_points)):
      if drawn_points[i-1] is None or drawn_points[i] is None:
        #Skip loop
        continue
      # Draw blue line on user interface
      cv2.line(frame, drawn_points[i-1], drawn_points[i], (255,0,0), 2)
      # Draw white line on blackboard for prediction
      cv2.line(blackboard, drawn_points[i-1], drawn_points[i], (255,255,255), 8)

    # User display text
    cv2.putText(
      frame, 
      "Multilayer Perceptron Prediction: {}".format(config.LETTERHASH[int(mlp_prediction + 1)]), 
      (10, 400), 
      cv2.FONT_HERSHEY_TRIPLEX, 
      0.9, 
      (255, 255, 255), 
      3
    )
    cv2.putText(
      frame, 
      "CNN Prediction: {}".format(config.LETTERHASH[int(cnn_prediction + 1)]), 
      (10, 430), 
      cv2.FONT_HERSHEY_TRIPLEX, 
      0.9, 
      (255, 255, 255), 
      3
    )

    # Show user webcam interface
    cv2.imshow("Real Time Letter Recognition", frame)

    # If the 'q' key is pressed, stop the primary frame detection loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  camera.release()
  cv2.destroyAllWindows()

# Code entry point
if __name__ == main():
  main()