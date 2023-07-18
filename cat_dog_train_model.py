# Import required modules
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def get_data(file_dir, TRAIN='train', VAL='val', TEST='test'):
    print("[INFO] Loading data...")
    # Initialize data transformation
    data_transform = {
        TRAIN : transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
        ]),
        VAL : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
        ]),
        TEST : transforms.Compose([
                transforms.Resize(254),
                transforms.CenterCrop(224),
                transforms.ToTensor()
        ])
    }
    # Initilize datasets and transform it
    datasets_img = {
        file : datasets.ImageFolder(
            os.path.join(file_dir, file),
            transform=data_transform[file]
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Load data into the desired format
    dataloaders = {
        file : torch.utils.data.DataLoader(
            datasets_img[file],
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Show result
    class_names = datasets_img[TRAIN].classes
    datasets_size = {file : len(datasets_img[file]) for file in [TRAIN, VAL, TEST]}
    for file in [TRAIN, VAL, TEST]:
        print(f"[INFO] Loaded {datasets_size[file]} images under {file}")
    print(f"Classes: {class_names}")
    
    return datasets_img, datasets_size, dataloaders, class_names

def get_vgg16_pretrained_model(weights):
    return None

def eval_model(vgg, criterion):
    return None

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    return None

# Main

## Use GPU if available
use_gpu = torch.cuda.is_available()
print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
## Define file directories
file_dir = './data'
output_dir = './output'
TRAIN = 'train' 
VAL = 'val'
TEST = 'test'
## Get Data
datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir, TRAIN, VAL, TEST)

