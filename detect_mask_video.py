# python detect_mask_video.py
# @author: Paulo Medeiros

# Import the used packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Take the frame's dimensions and use them to make a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Get the face detections by passing the blob through the network
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Build a list of faces, their associated positions, and the predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# Iterate through the face detections.
	for i in range(0, detections.shape[2]):
		# Filter out false positives by ensuring that the confidence level is higher than the minimum confidence level
		confidence = detections[0, 0, i, 2]

		# Exclude detections that aren't so good (false positives)
		if confidence > args["confidence"]:
			# Calculate the object's bounding box's (x, y) coordinate
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Make sure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract the face ROI, change the channel ordering from BGR to RGB, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) ### color
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Create separate lists for the face and bounding boxes
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Just make predictions if at least one face is detected
	if len(faces) > 0:
		# Instead of making one-by-one predictions in the for loop above,
		# we'll make batch predictions on all faces at the same time
		# for faster inference.

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Return a two-tuple containing the face positions and their corresponding locations
	return (locs, preds)

# Build an argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# From hard drive, load our serialized face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the model for the face mask detector from hard drive
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Start the video stream and give the camera sensor a chance to warm up.
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Iterate through the frames of the video stream
while True:
	# Take a frame from the threaded video stream and resize it to a width of 400 pixels maximum
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Identify faces in the photo and decide whether or not they are wearing a mask
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Go through all of the detected face positions and their corresponding locations in a loop
	for (box, pred) in zip(locs, preds):
		# The bounding box and prediction must be unpacked
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Choose a color for the class and a sticker for it (green for mask and red for no mask)
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Show the confidence (probability) in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# On the output frame, show the mark and box rectangle
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Display the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# If `q` key is pressed, break and exit frame
	if key == ord("q"):
		break

# Shut down all windows and switch off the video stream program. 
cv2.destroyAllWindows()
vs.stop()
