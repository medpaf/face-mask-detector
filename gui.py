# @author: Paulo Medeiros

# Import used packages
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

# Declare global variables
global filename
global fileLabel
global btnScanImg
global folderLabel
global datasetPath

# Create each function that will be performed after a button click

def train_model():
    global datasetPath
    
    if os.path.exists('config/dataset.txt'):
        # If OS is Linux
        if os.name == 'posix':
            os.system('python3 train_mask_detector.py --dataset ' + datasetPath)
        # If OS is Windows or macOS
        else:
            os.system(f'python train_mask_detector.py --dataset "{datasetPath}"')
    else:
        messagebox.showerror('Error', 'Dataset path is not specified. Please input the path in the Settings tab.')

def select_folder_to_database():
    global folderLabel
    global datasetPath

    datasetFolder = filedialog.askdirectory(initialdir='/', title='Select the database folder')

    #folderLabel.destroy()
    folderLabel = ttk.Label(tabConf, text=datasetFolder)
    folderLabel.pack(padx=10, pady=(0, 10))

    # Save the folder path to a .txt file so that it could be read the next time the app opens
    file = open('config/dataset.txt', 'w+')
    file.write(datasetFolder)
    datasetPath = datasetFolder
    file.close()

def select_img_to_scan():
    global filename
    global fileLabel
    global btnScanImg

    filename = filedialog.askopenfilename(initialdir='/', title='Select an image to scan', filetypes=(('jpg files', '*.jpg'), ('All files', '*.*')))
    fileLabel = ttk.Label(tabImg)
    fileLabel.pack(padx=10, pady=(0, 10))
    fileLabel.config(text=filename)

    btnScanImg = ttk.Button(tabImg)

    btnScanImg.config(text='Start', command=perform_img_scan)
    btnScanImg.pack(padx=10, pady=(0, 10))

def perform_img_scan():
    global filename

    fileLabel.destroy()
    btnScanImg.destroy()
    # If OS is Linux
    if os.name == 'posix':
        os.system('python3 detect_mask_image.py --image ' + filename) 
    # If OS is Windows or macOS
    else:
        os.system(f'python detect_mask_image.py --image "{filename}"')

def perform_video_scan():
    # If OS is Linux
    if os.name == 'posix':
        os.system('python3 detect_mask_video.py')
    # If OS is Windows or macOS
    else:
        os.system('python detect_mask_video.py')

# Create root widget window
root = tk.Tk()
root.title("Face Mask Detector GUI")
tabControl = ttk.Notebook(root)
root.configure()

# Create all the necessary tabs
tabTrain = ttk.Frame(tabControl)
tabImg = ttk.Frame(tabControl)
tabVid = ttk.Frame(tabControl)
tabConf = ttk.Frame(tabControl)
tabAbout = ttk.Frame(tabControl)

tabControl.add(tabTrain, text='Training')
tabControl.add(tabImg, text='Image')
tabControl.add(tabVid, text='Video')
tabControl.add(tabConf, text='Settings')
tabControl.add(tabAbout, text='About')
tabControl.pack(expand=1, fill='both', padx=15, pady=15)

# ADD ELEMENTS TO THE TRAINING TAB
TrainTabLabel = 'This is the training tab. Use this if you want to train or retrain the model.'
tabTrainLabel = ttk.Label(tabTrain, text=TrainTabLabel)
tabTrainLabel.pack(padx=10, pady=(10, 0))

btnTrain = ttk.Button(tabTrain, text='Train model', command=train_model)
btnTrain.pack(padx=10, pady=10)

# ADD ELEMENTS TO THE IMAGE TAB
imgTabLabel = 'Use this tab to perform detection on static images.'
tabImgLabel = ttk.Label(tabImg, text=imgTabLabel)
tabImgLabel.pack(padx=10, pady=(10, 0))

btnAddImg = ttk.Button(tabImg, text='Browse', command=select_img_to_scan)
btnAddImg.pack(padx=10, pady=10)

fileLabel = ttk.Label(tabImg)
fileLabel.pack(padx=10, pady=(0, 10))

btnScanImg = ttk.Button(tabImg)

# ADD ELEMENTS TO THE VIDEO TAB
VidTabLabel = 'This is the video tab. Click the button to start the webcam detector program.'
tabVidLabel = ttk.Label(tabVid, text=VidTabLabel)
tabVidLabel.pack(padx=10, pady=(10, 0))

btnVid = ttk.Button(tabVid, text='Start', command=perform_video_scan)
btnVid.pack(padx=10, pady=10)

# ADD ELEMENTS TO THE SETTINGS TAB
confTabLabel = 'Use this tab to make configurations.'
tabConfLabel = ttk.Label(tabConf, text=confTabLabel)
tabConfLabel.pack(padx=10, pady=10)

tabSetDatabase = ttk.Label(tabConf, text='Path to the dataset')
tabSetDatabase.pack(padx=10, pady=(10, 0))

btnSetDb = ttk.Button(tabConf, text='Browse', command=select_folder_to_database)
btnSetDb.pack(padx=10, pady=10)

# ADD ELEMENTS TO THE ABOUT TAB
confAboutLabel = 'Developed by Paulo Medeiros.'
tabAboutLabel = ttk.Label(tabAbout, text=confAboutLabel)
tabAboutLabel.pack(padx=10, pady=10)

# Check if there is file in the folder path
if os.path.exists('config/dataset.txt'):
    file = open('config/dataset.txt', 'r')
    datasetPath = file.read()

    folderLabel = ttk.Label(tabConf, text=datasetPath)
    folderLabel.pack(padx=10, pady=(0, 10))

# Loop to display the root window
root.mainloop()
