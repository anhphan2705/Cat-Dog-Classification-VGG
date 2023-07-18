import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
# import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy

## Define file directories
file_dir = './data'
output_dir = './output/VGG16_trained.pth'
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

def eval_model(vgg, criterion, dataset=VAL):
    print('-'*18)
    print("[Evaluation Model] Evaluating...")
    # Keeping track
    since = time.time()
    avg_loss = 0
    avg_accuracy = 0
    loss_test = 0
    accuracy_test = 0
    
    batches = len(dataloaders[dataset])
    # 1 forward pass each data without calculating gradient to find the prediction before training
    for i, data in enumerate(dataloaders[dataset]):
        print("\r[Evaluation Model] Evaluate '{}' batch {}/{}".format(dataset, i+1, batches), end='', flush=True)
            
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
    
    avg_loss = loss_test / datasets_size[dataset]
    avg_accuracy = accuracy_test / datasets_size[dataset]
    
    elapsed_time = time.time() - since
    print()
    print(f"[Evaluation Model] Evaluation completed in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    print(f"[Evaluation Model] Avg loss      ({dataset}): {avg_loss:.4f}")
    print(f"[Evaluation Model] Avg accuracy  ({dataset}): {avg_accuracy:.4f}")
    print('-'*18)
    return avg_loss, avg_accuracy

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    print('\n','#'*15, ' TRAINING ', '#'*15, '\n')
    print('[TRAIN MODEL] Training...')
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_accuracy = 0.0
    avg_loss = 0
    avg_accuracy = 0
    avg_loss_val = 0
    avg_accuracy_val = 0
    
    train_batches = len(dataloaders[TRAIN])
    
    for epoch in range(num_epochs):
        print('')
        print(f"[TRAIN MODEL] Epoch {epoch+1}/{num_epochs}")
        loss_train = 0
        accuracy_train = 0
        vgg.train(True)
        
        for i, data in enumerate(dataloaders[TRAIN]):
            print("\r[TRAIN MODEL] Training batch {}/{}".format(i+1, train_batches // 2), end='', flush=True)
            # Use half training dataset
            if i >= train_batches // 2:
                break
            inputs, labels = data
            # Forward pass
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)   
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            
            # Back propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Save result
            loss_train += loss.data
            accuracy_train += torch.sum(preds == labels.data)
            
            # Clear cache
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
            
        avg_loss = loss_train * 2 / datasets_size[TRAIN]
        avg_accuracy = accuracy_train * 2 /datasets_size[TRAIN]
        vgg.train(False)
        vgg.eval()
        print('')
        avg_loss_val, avg_accuracy_val = eval_model(vgg, criterion, dataset=VAL)
        
        print('-'*13)
        print(f"[TRAIN MODEL] Epoch {epoch+1} result: ")
        print(f"[TRAIN MODEL] Avg loss      (train):    {avg_loss:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy  (train):    {avg_accuracy:.4f}")
        print(f"[TRAIN MODEL] Avg loss      (val):      {avg_loss_val:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy  (val):      {avg_accuracy_val:.4f}")
        print('-'*13)
        
        if avg_accuracy_val > best_accuracy:
            best_accuracy = avg_accuracy_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print(f"[TRAIN MODEL] Training completed in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    print(f"[TRAIN MODEL] Best accuracy: {best_accuracy:.4f}")
    print('\n','#'*15, ' FINISHED ', '#'*15, '\n')
    vgg.load_state_dict(best_model_wts)
    return vgg

if __name__ ==  '__main__':
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir, TRAIN, VAL, TEST)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_pretrained_model(len_target=2)
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Define model requirements
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=1)
    # Evaluate before training
    print("[INFO] Before training evalutaion in progress...")
    eval_model(vgg16, criterion, dataset=TEST)
    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
    torch.save(vgg16.state_dict(), output_dir)
    print("[INFO] After training evalutaion in progress...")
    eval_model(vgg16, criterion, dataset=TEST)
    