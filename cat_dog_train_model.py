import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

## Define file directories
file_dir = './data'
output_dir = './output'
TRAIN = 'train' 
VAL = 'val'
TEST = 'test'

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

def get_vgg16_pretrained_model(model_dir='', weights=models.VGG16_BN_Weights.DEFAULT, len_target=1000):
    print("[INFO] Getting VGG-16 pre-trained model...")
    # Load VGG-16 pretrained model (1000 features)
    if model_dir == '':
        vgg16 = models.vgg16_bn(weights)
    else: 
        vgg16.load_state_dict(torch.load(model_dir))
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.requires_grad = False
    # Get feature quantity of the last layer
    num_features = vgg16.classifier[-1].in_features
    # Remove the last layer
    features = list(vgg16.classifier.children())[:-1]
    # Add custom layer with custom outputs
    features.extend([nn.Linear(num_features, len_target)])
    # Replace the model classifier
    vgg16.classifier = nn.Sequential(*features)
    print("[INFO] Loaded VGG-16 pre-trained model\n", vgg16, "\n")
    return vgg16

def eval_model(vgg, criterion):
    print('-'*18)
    print("[Evaluation Model] Evaluating...")
    # Keeping track
    since = time.time()
    avg_loss = 0
    avg_accuracy = 0
    loss_test = 0
    accuracy_test = 0
    
    test_batches = len(dataloaders[TEST])
    
    # 1 forward pass each data without calculating gradient to find the prediction before training
    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\r[Evaluation Model] Test batch {}/{}".format(i, test_batches), end='', flush=True)
            
        vgg.train(False)
        vgg.eval()
        inputs, labels = data
        
        with torch.no_grad():
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
                
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        loss_test += loss.data
        accuracy_test += torch.sum(preds == labels.data)
        
        # Clear cache to prevent out of memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    
    avg_loss = loss_test / datasets_size[TEST]
    avg_accuracy = accuracy_test / datasets_size[TEST]
    
    elapsed_time = time.time() - since
    print("[Evaluation Model] Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("[Evaluation Model] Avg loss (test): {:.4f}".format(avg_loss))
    print("[Evaluation Model] Avg accuracy (test): {:.4f}".format(avg_accuracy))
    print('-'*18)

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    return None

if __name__ ==  '__main__':
    # Main

    ## Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    ## Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir, TRAIN, VAL, TEST)
    ## Get VGG16 pre-trained model
    vgg16 = get_vgg16_pretrained_model(len_target=2)
    ## Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    ## Define model requirements
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=1)
    ## Test before training
    print("[INFO] Before training evalutaion in progress...")
    eval_model(vgg16, criterion)