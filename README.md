# Face mask detector

Computer vision and machine learning application implemented in Python to detect masks in one or more faces.
Supports detection from static images and video streams from webcam.

The easiest way to use this program is to run the gui.py file (currently, only supported on UNIX systems). If you are using the Windows operating systems run the code (`detect_mask_image.py` or `detect_mask_video.py`) directly from the terminal.

The dependencies to run this script are:

  1) Tensorflow
  2) Keras
  3) OpenCV (cv2)
  4) Skicit-learn (sklearn)
  5) Matplotlib
  6) Numpy
  7) Imutils
  8) Tkinter
  
## Introduction

Everyone recognizes the importance of wearing a mask to reduce the spread of SARS-CoV-2, the virus that causes COVID-19. However, instead of letting people
monitor each other, the intention of this project is to develop a computer program that would be able to analyze images or real-time video stream from a webcam and detect whether people are wearing a mask or not.
This mask recognition software does not hurt privacy because the program does not actually identify people. The software is trained according to two sets of images: one that teaches the algorithm how to recognize a face (“face detection”) and another that
defines how to recognize a face with a mask (“mask recognition”). The machine learning algorithm does not identify the face in a way that can link it to a specific person, because it does not use any type of training that links faces to identities.

This project consists of four Python scripts:
- `train_mask_detector.py`: Takes the input dataset and creates the detector model using the MobileNetV2 convolutional neural network architecture. A training report image containing the accuracy/loss graph is also created when this script is run.
- `detect_mask_image.py`: Search for facial masks in static images.
- `detect_mask_video.py`: Search for facial masks in every frame of the webcam video stream.
- `gui.py`: Combines all three previously cited scripts into one graphical user interface program.

## Installation

You'll need to install the necessary packages so that the script can run withouth any problems.

### Linux

Before installing the dependencies, please make sure you have `python3` installed on your machine. But since almost all Linux distros come with Python pre-installed you probably won't need to perform this step. After that, on the Linux terminal, type the following commands as root:
```
sudo pip3 install tensorflow
sudo pip3 install keras
sudo pip3 install sklearn
sudo pip3 install cv2
sudo pip3 install matplotlib
sudo pip3 install numpy
sudo pip3 install imutils
sudo pip3 install tk
```

### Windows

Since Windows doesn't come with Python pre-installed, you'll need to install it if you haven't already. Follow the link below to download Python.

[Python](https://www.python.org/downloads/windows/) (it is recommended to install the stable release)

After thar, install the following dependecies typing the following commands on the Command Prompt (CMD):
```
python -m pip install tensorflow
python -m pip install keras
python -m pip install sklearn
python -m pip install cv2
python -m pip install matplotlib
python -m pip install numpy
python -m pip install imutils
python -m pip install tk
```

## How to use it

### Training the model

To train the model, the `gui.py` file can be used or the following command can be issued on the terminal (inside the project’s root folder):

`python3 train_mask_detector.py --dataset dataset`

After issuing this command, the terminal will continuously prompt current information about the training process including accuracy, loss, number of epoch and
remaining time.

![train](https://user-images.githubusercontent.com/61552222/134815701-8dcf7de2-e064-49e6-8051-77a6127101e7.png)

After the training process is done, inside the root folder, an image file will be created containing the performance metrics of the model. Also, some metrics like precision, recall f1-score and support will be displayed in the terminal.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61552222/134815737-239fcf18-df6f-4498-8495-8ffa574aa492.png" />
</p>

### Face mask detection for static images

To test the model using static images, apart from using the GUI, which is recommended since it makes the process fairly simple, the following command can be
issued in the terminal as an example:

`python3 detect_mask_image.py --image examples/ex1`

More images were then used to continue to test the model. Some of the results of these tests are shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61552222/134815850-9493c9f7-354b-4e9e-9740-ccd988ed5725.png" />  
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61552222/134815851-f4a3c1c9-abed-4512-973e-c8ee2999b04d.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61552222/134815853-321e27b7-6daa-4a3c-a72f-a9354374be95.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61552222/134815856-19935b8f-967b-4a4d-815e-6e5118e39a80.png" />
</p>

### Face mask detection for real-time video 

After assuring that the detect_mask_image.py script is working properly, it is time to test the detect_mask_video.py script to see if it also preforms without issues. Once again, to do that the `gui.py` file can be used or the following command can be issued in the terminal:

`python3 detect_mask_video.py`

<p align="center">
 <img src= https://user-images.githubusercontent.com/61552222/134815890-2d36ea22-045f-48ff-b166-33a99026dbf2.png />
</p>

