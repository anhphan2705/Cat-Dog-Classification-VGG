import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy

## Define file directories
file_dir = './data'
output_dir = './output/VGG16_trained.pth'
TRAIN = 'train' 
VAL = 'val'
TEST = 'test'

def get_data(file_dir):
    """
    Load and transform the data using PyTorch's ImageFolder and DataLoader.

    Args:
        file_dir (str): Directory path containing the data.
        TRAIN (str, optional): Name of the training dataset directory. Defaults to 'train'.
        VAL (str, optional): Name of the validation dataset directory. Defaults to 'val'.
        TEST (str, optional): Name of the test dataset directory. Defaults to 'test'.

    Returns:
        datasets_img (dict): Dictionary containing the datasets for training, validation, and test.
        datasets_size (dict): Dictionary containing the sizes of the datasets.
        dataloaders (dict): Dictionary containing the data loaders for training, validation, and test.
        class_names (list): List of class names.
    """
    print("[INFO] Loading data...")
    # Initialize data transformations
    data_transform = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        TEST: transforms.Compose([
            transforms.Resize(254),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }
    # Initialize datasets and apply transformations
    datasets_img = {
        file: datasets.ImageFolder(
            os.path.join(file_dir, file),
            transform=data_transform[file]
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Load data into dataloaders
    dataloaders = {
        file: torch.utils.data.DataLoader(
            datasets_img[file],
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        for file in [TRAIN, VAL, TEST]
    }
    # Get class names and dataset sizes
    class_names = datasets_img[TRAIN].classes
    datasets_size = {file: len(datasets_img[file]) for file in [TRAIN, VAL, TEST]}
    for file in [TRAIN, VAL, TEST]:
        print(f"[INFO] Loaded {datasets_size[file]} images under {file}")
    print(f"Classes: {class_names}")

    return datasets_img, datasets_size, dataloaders, class_names


def get_vgg16_pretrained_model(model_dir='', weights=models.VGG16_BN_Weights.DEFAULT, len_target=1000):
    """
    Retrieve the VGG-16 pre-trained model and modify its classifier for the desired number of output classes.

    Args:
        model_dir (str, optional): Directory path for loading a pre-trained model state dictionary. Defaults to ''.
        weights (str or dict, optional): Pre-trained model weights. Defaults to models.vgg16_bn(pretrained=True).state_dict().
        len_target (int, optional): Number of output classes. Defaults to 1000.

    Returns:
        vgg16 (torchvision.models.vgg16): VGG-16 model with modified classifier.
    """
    print("[INFO] Getting VGG-16 pre-trained model...")
    # Load VGG-16 pretrained model
    vgg16 = models.vgg16_bn(weights)
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.requires_grad = False
    # Get number of features in the last layer
    num_features = vgg16.classifier[-1].in_features
    # Remove the last layer
    features = list(vgg16.classifier.children())[:-1]
    # Add custom layer with custom number of output classes
    features.extend([nn.Linear(num_features, len_target)])
    # Replace the model's classifier
    vgg16.classifier = nn.Sequential(*features)
    
    # If load personal pre-trained model
    if model_dir != '':
        vgg16.load_state_dict(torch.load(model_dir))
    print("[INFO] Loaded VGG-16 pre-trained model\n", vgg16, "\n")
    
    return vgg16


def eval_model(vgg, criterion, dataset=VAL):
    """
    Evaluate the model's performance on the specified dataset.

    Args:
        vgg (torchvision.models.vgg16): Model to evaluate.
        criterion (torch.nn.modules.loss): Loss function.
        dataset (str, optional): Dataset to evaluate. Defaults to 'val'.

    Returns:
        avg_loss (float): Average loss on the dataset.
        avg_accuracy (float): Average accuracy on the dataset.
    """
    print('-' * 18)
    print("[Evaluation Model] Evaluating...")
    since = time.time()
    avg_loss = 0
    avg_accuracy = 0
    loss_test = 0
    accuracy_test = 0

    batches = len(dataloaders[dataset])
    # Perform forward pass on the dataset
    for i, data in enumerate(dataloaders[dataset]):
        print(f"\r[Evaluation Model] Evaluate '{dataset}' batch {i + 1}/{batches} ({len(data[1])*(i+1)} images)", end='', flush=True)

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
    print('-' * 18)
    return avg_loss, avg_accuracy


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    """
    Train the model using the training dataset and evaluate its performance on the validation dataset.

    Args:
        vgg (torchvision.models.vgg16): Model to train.
        criterion (torch.nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer for model parameter updates.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.

    Returns:
        vgg (torchvision.models.vgg16): Trained model.
    """
    print('\n', '#' * 15, ' TRAINING ', '#' * 15, '\n')
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
        print(f"[TRAIN MODEL] Epoch {epoch + 1}/{num_epochs}")
        loss_train = 0
        accuracy_train = 0
        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
            print(f"\r[TRAIN MODEL] Training batch {i + 1}/{train_batches} ({len(data[1])*(i+1)} images)", end='', flush=True)
            # Use only half of the training dataset
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

            # Backward propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save results
            loss_train += loss.data
            accuracy_train += torch.sum(preds == labels.data)

            # Clear cache
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train * 2 / datasets_size[TRAIN]
        avg_accuracy = accuracy_train * 2 / datasets_size[TRAIN]
        vgg.train(False)
        vgg.eval()
        print('')
        avg_loss_val, avg_accuracy_val = eval_model(vgg, criterion, dataset=VAL)

        print('-' * 13)
        print(f"[TRAIN MODEL] Epoch {epoch + 1} result: ")
        print(f"[TRAIN MODEL] Avg loss      (train):    {avg_loss:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy  (train):    {avg_accuracy:.4f}")
        print(f"[TRAIN MODEL] Avg loss      (val):      {avg_loss_val:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy  (val):      {avg_accuracy_val:.4f}")
        print('-' * 13)

        if avg_accuracy_val > best_accuracy:
            best_accuracy = avg_accuracy_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print(f"[TRAIN MODEL] Training completed in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    print(f"[TRAIN MODEL] Best accuracy: {best_accuracy:.4f}")
    print('\n', '#' * 15, ' FINISHED ', '#' * 15, '\n')
    vgg.load_state_dict(best_model_wts)
    return vgg


if __name__ == '__main__':
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_pretrained_model(len_target=2)
    # vgg16 = get_vgg16_pretrained_model('./output/VGG16_trained.pth', len_target=2)      # If load custom pre-trained model, watch out to match len target
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Define model requirements
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=1)
    # Evaluate before training
    print("[INFO] Before training evaluation in progress...")
    eval_model(vgg16, criterion, dataset=TEST)
    # Training
    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
    torch.save(vgg16.state_dict(), output_dir)
    # Evaluate after training
    print("[INFO] After training evaluation in progress...")
    eval_model(vgg16, criterion, dataset=TEST)