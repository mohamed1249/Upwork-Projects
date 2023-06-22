# Binary classifier training #

This repository contains a notebook, training file and requirements.txt file to enable training of a binary image classifier.

The classifier can be trained from scratch or be based on pretrained models from the Pytorch model zoo.

Training data is required in image form, preferably jpg format.
For training, the data needs to be divided up into training set and validation set in the following order:
```
└── data
    ├── train
    │   ├── class_1
    │   └── class_2
    └── val
        ├── class_1
        └── class_2
```
With all data in only 2 Folders, I suggest using these as the "train" folders and creating the "val" folders and
moving a randomly chosen subset of images there with

```bash
shuf -zn<number of images(no space before)> -e *.jpg | xargs -0 -I{} mv -v {} target/
```
from within each class folder.

To use the trainer with a docker container on a cloud instance, the absolute path of the data folder must
be given to the container at runtime (shown below).

To save models to the instance, the absolute path of the desired model folder must be given to
the container at runtime (shown below).

Pretrained models will be downloaded as required.

# Training with Docker #

After cloning the repository, build the container with the command
```bash
docker build . -t binary_class:latest
```

An image with the name "binary_class:latest" should appear in the list of images
```bash
docker image list
```

The container can be run with 
```bash
 docker run --gpus all -v <path/to/data/folder>:/binary_class/data -v <path/to/model/folder>:/binary_class/models -it binary_class:latest bash
```
with the paths in absolute form as explained above.

This command should result in a shell inside the container. The classifier can be trained with

```bash
python train.py
```
with optional arguments
```bash
  -h, --help                show this help message and exit
  --data_dir DATA_DIR       data directory
  --model_dir MODEL_DIR     model directory
  --num_epochs NUM_EPOCHS   number of epochs for training
  --batch_size BATCH_SIZE   batch size for training
  --step_size STEP_SIZE     number of steps after which to reduce learning rate
  --gamma GAMMA             factor to multiply learning rate
  --lr LR                   learning rate
```