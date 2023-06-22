# How to use and run the YOLOv5 Python Code
This documentation explains how to use and run the YOLOv5 Python code to train and detect objects in images.
### Installation
First, clone the YOLOv5 repository by running the following command in your terminal:
`!git clone https://github.com/ultralytics/yolov5`
After cloning the repository, navigate to the cloned directory by running:
`%cd yolov5`
Then, install the necessary dependencies for the yolov5 repository using pip:
`!pip install -U -r yolov5/requirements.txt `

## Creating data.yaml file
creat a file with the name data.yaml and include this in it:
```
train: /path/to/data/training/images
val: /path/to/data/validation/images

train_labels: /path/to/data/training/labels
val_labels: /path/to/data/validation/labels

nc: 20 # number of classes
names: [
'Cabin Cruiser', 'Canoe/Kayak',
'Commercial', 'Dinghies/Cats',
'Half Cab', 'Hire & Drive',
'House Boat', 'Human', 'Kite',
'Open', 'Other', 'PFD (lifejacket) ', 'PWC',
'Registration Number',
'Rowing', 'SUP', 'Ski Boat',
'Structure', 'Windsurfer',
'Yacht'] # classes names
```

## Training
To train the model, run the following command:

`!python train.py --img 640 --batch 32 --epochs 100 --data path/to/data.yaml --cfg models/yolov5s.yaml --weights models/yolov5s.pt --name yolo_model --nosave --cache`

This command trains the YOLOv5s model for 100 epochs using 32 images per batch and image size of 640x640. It saves the trained model in the yolov5/runs/train/yolo_model directory.
## Validation
To validate the trained model, run the following command:

`!python val.py --data path/to/data.yaml --weights path/to/last.pt`

This command evaluates the performance of the trained model on the validation set and displays the precision and recall values.
## Object Detection
To detect objects in a set of images, you can use the following code:
```
import os   # Import the os module to execute shell commands
import time # Import the time module to pause the execution of the script

for img in os.listdir(r"New_DataSet\New_DataSet"):
    os.popen(fr'python yolov5/detect.py --source "path\to\{img}" --weights paht/to/last.pt --conf 0.1')
    time.sleep(6)
    print(img, ' Done!')
```
This code detects objects in all images in the New_DataSet\New_DataSet directory using the pre-trained model last.pt. It pauses for 6 seconds between each image to allow time for the model to detect objects. The detected objects are displayed in the terminal.

To detect objects in a set of images and save the results in a text file, use the following code:
```
import os   # Import the os module to execute shell commands
import time # Import the time module to pause the execution of the script

for img in os.listdir(r"images"):
    os.popen(fr'python yolov5/detect_txt.py --source "path\to\{img}" --weights path\to\last.pt --conf 0.1 --save-txt --save-conf')
    time.sleep(6)
    print(img, ' Done!')
```
This code detects objects in all images in the images directory using the pre-trained model last.pt and saves the results in a text file.
### Visualizing Results
To visualize the results of object detection, use the following code:
```
import os   # Import the os module to execute shell commands
import time # Import the time module to pause the execution of the script

for img in os.listdir(r"images"):
    os.popen(fr'python yolov5/detect_txt.py --source "images\{img}" --weights last.pt --conf 0.1 --save-txt --save-conf')
    time.sleep(6)
    print(img, ' Done!')
```
This code detects objects in all images in the images directory using the pre-trained model last.pt and saves the results in a text file.

### Visualizing Results
To visualize the results of object detection, use the following code:
``` 
import os   # Import the os module to execute shell commands
import time # Import the time module to pause the execution of the script
import matplotlib.pyplot as plt

n = 0

for d in os.listdir(r'yolov5\runs\detect'):
    for i in os.listdir(os.path.join(r'yolov5\runs\detect',d)):
        if n < 10:
            img = plt.imread(os.path.join(r'yolov5\runs\detect',d,i))
            plt.imshow(img)
            plt.show()
            n += 1
        else
```