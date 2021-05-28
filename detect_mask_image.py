# python3 detect_mask_image.py --image examples/example_01.png OR python3 detect_mask_image.py -i examples/example_01.png
# @author: Paulo Medeiros

# Import used packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Create a parser for arguments and parse them.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Get the face detector model from the hard drive.
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Get the face mask detector model from the hard drive.
print("[INFO] Loading face mask detector model...")
model = load_model(args["model"])

# Load the input image from hard drive, copy it, and get the spatial dimensions of the image
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# Make a blob out of the picture
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Get the face detections by passing the blob through the network.
print("[INFO] Computing face detections...")
net.setInput(blob)
detections = net.forward()

# Iterate through the face detections.
for i in range(0, detections.shape[2]):
	# Get the likelihood (or confidence) [probability] associated with the detection.
	confidence = detections[0, 0, i, 2]

	# Filter out false positives by ensuring that the confidence level is higher than the minimum confidence level.
	if confidence > args["confidence"]:
		# Calculate the cartesian coordinates (x, y) of the object's bounding box.
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Make sure the bounding boxes are within the frame's dimensions.
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# Select the face ROI, adjust the channel ordering from BGR to RGB, and resize it to 224x224 pixels., and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# Run the face through the model to see whether it has a mask on it or not.
		(mask, withoutMask) = model.predict(face)[0]

		# Choose a class mark and a color to use for the bounding box and text.
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Show the confidence (probability) in the label.
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# On the output panel, show the mark and bounding box rectangle.
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Display the final picture
cv2.imshow("Output", image)
cv2.waitKey(0)
