import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import copy


## Define file directories
file_dir = './data-shorten'
out_model_dir = './output/VGG16_trained.pth'
out_plot_dir = './output/epoch_progress.jpg'
out_report_dir = './output/classification_report.txt'
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


def get_epoch_progress_graph(accuracy_train, loss_train, accuracy_val, loss_val, save_dir=out_plot_dir):
    """
    Plot the progress of accuracy and loss during training epochs.

    Args:
        accuracy_train (list): List of accuracy values for training set at each epoch.
        loss_train (list): List of loss values for training set at each epoch.
        accuracy_val (list): List of accuracy values for validation set at each epoch.
        loss_val (list): List of loss values for validation set at each epoch.
        save_dir (str): Directory path to save the plot image. Defaults to './output/epoch_progress.jpg'.
    """
    print("[PLOT] Getting plot...")
    # Main window
    fig = plt.figure(figsize =(20, 10))
    sub1 = plt.subplot(2, 1, 1)
    sub2 = plt.subplot(2, 1, 2)
    
    # Subplot 1: Epoch vs Accuracy
    sub1.plot(accuracy_train, linestyle='solid', color='r')
    sub1.plot(accuracy_val, linestyle='solid', color='g')
    sub1.set_xticks(list(range(0, len(accuracy_train)+3)))
    sub1.legend(labels=["train", "val"], loc='best')
    sub1.plot(accuracy_train, 'or')
    sub1.plot(accuracy_val, 'og')
    sub1.set_xlabel("Epoch")
    sub1.set_ylabel("Accuracy")
    sub1.set_title("Epoch Accuracy")
    
    # Subplot 2: Epoch vs Loss
    sub2.plot(loss_train, linestyle='solid', color='r')
    sub2.plot(loss_val, linestyle='solid', color='g')
    sub2.set_xticks(list(range(0, len(loss_train)+3)))
    sub2.legend(labels=["Train", "Val"], loc='best')
    sub2.plot(loss_train, 'or')
    sub2.plot(loss_val, 'og')
    sub2.set_xlabel("Epoch")
    sub2.set_ylabel("Loss")
    sub2.set_title("Epoch Loss")
    
    # Output
    print("[PLOT] Outputing plot...")
    plt.savefig(save_dir)
    plt.show()
    

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
        vgg16.eval()
    print("[INFO] Loaded VGG-16 pre-trained model\n", vgg16, "\n")
    
    return vgg16


def get_classification_report(truth_values, pred_values):
    """
    Generate a classification report and confusion matrix based on ground truth and predicted labels.

    Args:
        truth_values (list): List of ground truth labels.
        pred_values (list): List of predicted labels.

    Returns:
        None
    """
    report = classification_report(truth_values, pred_values, target_names=class_names,  digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all') 
    print('[Evalutaion Model] Showing detailed report\n')
    print(report)
    print('[Evalutaion Model] Showing confusion matrix')
    print(f'                       Predicted Label              ')
    print(f'                         0            1         ')
    print(f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}')
    print(f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}')
    
    
def save_classification_report(truth_values, pred_values, out_report_dir):
    """
    Save the classification report and confusion matrix to a text file.

    Args:
        truth_values (list): List of ground truth labels.
        pred_values (list): List of predicted labels.
        out_report_dir (str): Directory path to save the classification report file.
        
    Returns:
        None
    """
    print('[INFO] Saving report...')
    c_report = classification_report(truth_values, pred_values, target_names=class_names,  digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all') 
    matrix_report = ['                       Predicted Label              ', 
                     f'                         0            1         ',
                     f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}',
                     f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}']
    
    with open(out_report_dir, 'w') as f:
        f.write(c_report)
        f.write('\n')
        for line in matrix_report:
            f.write(line)
            f.write('\n')
        

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
    print('-' * 60)
    print("[Evaluation Model] Evaluating...")
    since = time.time()
    avg_loss = 0
    avg_accuracy = 0
    loss_test = 0
    accuracy_test = 0
    pred_values = []
    truth_values = []

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
        accuracy_test += torch.sum(preds == labels.data)
        loss_test += loss.data
        
        for i in range(len(preds)):
            pred_values.append(preds.cpu().numpy()[i])
            truth_values.append(labels.data.cpu().numpy()[i])

        # Clear cache to prevent out of memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / datasets_size[dataset]
    avg_accuracy = accuracy_test / datasets_size[dataset]

    elapsed_time = time.time() - since
    print()
    print(f"[Evaluation Model] Evaluation completed in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    print(f"[Evaluation Model] Avg loss         ({dataset}): {avg_loss:.4f}")
    print(f"[Evaluation Model] Avg accuracy     ({dataset}): {avg_accuracy:.4f}")
    get_classification_report(truth_values, pred_values)
    if dataset == TEST:
        save_classification_report(truth_values, pred_values, out_report_dir)
    print('-' * 60)
    return avg_loss, avg_accuracy


def train_model(vgg, criterion, optimizer, scheduler, dataset=TRAIN, num_epochs=10):
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
    losses = []
    accuracy = []
    losses_val = []
    accuracy_val = []

    train_batches = len(dataloaders[dataset])

    for epoch in range(num_epochs):
        print('')
        print(f"[TRAIN MODEL] Epoch {epoch + 1}/{num_epochs}")
        loss_train = 0
        accuracy_train = 0
        vgg.train(True)

        for i, data in enumerate(dataloaders[dataset]):
            print(f"\r[TRAIN MODEL] Training batch {i + 1}/{train_batches} ({len(data[1])*(i+1)} images)", end='', flush=True)
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
            scheduler.step()

            # Save results
            loss_train += loss.data
            accuracy_train += torch.sum(preds == labels.data)

            # Clear cache
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / datasets_size[dataset]
        avg_accuracy = accuracy_train / datasets_size[dataset]
        vgg.train(False)
        vgg.eval()
        print('')
        avg_loss_val, avg_accuracy_val = eval_model(vgg, criterion, dataset=VAL)
        
        # Save data to plot graph
        losses.append(avg_loss.cpu())
        accuracy.append(avg_accuracy.cpu())
        losses_val.append(avg_loss_val.cpu())
        accuracy_val.append(avg_accuracy_val.cpu())
        
        # Print result
        print('-' * 13)
        print(f"[TRAIN MODEL] Epoch {epoch + 1} result: ")
        print(f"[TRAIN MODEL] Avg loss          (train):    {avg_loss:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy      (train):    {avg_accuracy:.4f}")
        print(f"[TRAIN MODEL] Avg loss          (val):      {avg_loss_val:.4f}")
        print(f"[TRAIN MODEL] Avg accuracy      (val):      {avg_accuracy_val:.4f}")
        print('-' * 13)

        if avg_accuracy_val > best_accuracy:
            best_accuracy = avg_accuracy_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print(f"[TRAIN MODEL] Training completed in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    print(f"[TRAIN MODEL] Best accuracy: {best_accuracy:.4f}")
    print('\n', '#' * 15, ' FINISHED ', '#' * 15, '\n')
    vgg.load_state_dict(best_model_wts)
    # Print Graph
    get_epoch_progress_graph(accuracy, losses, accuracy_val, losses_val)
    return vgg


if __name__ == '__main__':
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_pretrained_model(len_target=2)
    # vgg16 = get_vgg16_pretrained_model(model_dir='./output/VGG16_trained.pth', len_target=2)      # If load custom pre-trained model, watch out to match len target
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Define model requirements
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=1e-2, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)
    # Evaluate before training
    print("[INFO] Before training evaluation in progress...")
    eval_model(vgg16, criterion, dataset=TEST)
    # Training
    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    torch.save(vgg16.state_dict(), out_model_dir)
    # Evaluate after training
    print("[INFO] After training evaluation in progress...")
    eval_model(vgg16, criterion, dataset=TEST)