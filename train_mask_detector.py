# python3 train_mask_detector.py --dataset dataset OR python3 train_mask_detector.py -d dataset
# @author: Paulo Medeiros

# Import used packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Create a parser for arguments and parse them.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Set the learning rate, the number of epochs to train for, and the batch size.
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Get a list of images from our dataset directory, then initialize the list of data (i.e., images).
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Iterate through the image paths
for imagePath in imagePaths:
	# From the filename, extract the class label.
	label = imagePath.split(os.path.sep)[-2]

	# Load and preprocess the input image (224x224).
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# Refresh the data and labels lists, as required.
	data.append(image)
	labels.append(label)

# Build NumPy arrays from the data and labels.
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Encode the labels in a single pass.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Divide the data into training and research segments, with 75% of the data going to training and 25% going to testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Build a data augmentation training image generator.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Load the MobileNetV2 network, making sure to switch off the head FC layer sets.
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Create the model's head, which will be put on top of the base model.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# On top of the base model, position the head FC model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop through all of the layers in the base model and freeze them so they don't get changed during the
# first training session.
for layer in baseModel.layers:
	layer.trainable = False

# Compile our model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network
print("[INFO] Training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Make predictions based on the data in the testing set
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Find the index of the mark with the highest expected likelihood for each picture in the testing set.
predIdxs = np.argmax(predIdxs, axis=1)

# Display a classification report that has been formatted.
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save the model as a serialized file.
print("[INFO] Saving mask detector model...")
model.save(args["model"], save_format="h5")

# Draw a graph with the training loss and precision.
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="training_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])