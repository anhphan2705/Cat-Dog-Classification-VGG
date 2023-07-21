# VGG-16 Image Classification Cat Dog

This repository contains a Python script for training and evaluating an image classification model based on the VGG-16 architecture using PyTorch. The model is capable of classifying images into two categories: dogs and cats.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- Scikit-learn

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/anhphan2705/Image-Classification-Dog-Cat.git
```

2. Install the required dependencies:

```bash
pip install torch torchvision matplotlib scikit-learn
```

3. Prepare the Data:

Place your training, validation, and test datasets in separate directories (`train`, `val`, `test`) inside the `data` directory as shown below:

```
Image-Classification-Dog-Cat/
  cat_dog_train_model.py
  pre-trained-model.pth (optional)
  data/
      train/
          class1/
              image1.jpg
              image2.jpg
              ...
          class2/
              image1.jpg
              image2.jpg
              ...
          ...
      val/
          class1/
              image1.jpg
              image2.jpg
              ...
          class2/
              image1.jpg
              image2.jpg
              ...
          ...
      test/
          class1/
              image1.jpg
              image2.jpg
              ...
          class2/
              image1.jpg
              image2.jpg
              ...
          ...
```

If you want to load a pre-trained model from your local computer, add the `model_dir` argument to the `get_vgg16_pretrained_model()` function in the `cat_dog_train_model.py` script.

4. Training and Evaluation:

   - Open the `cat_dog_train_model.py` file and modify the necessary parameters such as file directories, output directory, and your data files name.

     ```python
     file_dir = './data-shorten'
     out_model_dir = './output/VGG16_trained.pth'
     out_plot_dir = './output/epoch_progress.jpg'
     out_report_dir = './output/classification_report.txt'
     TRAIN = 'train' 
     VAL = 'val'
     TEST = 'test'
     ```
   - Run the script:

     ```bash
     python ./cat_dog_train_model.py
     ```

   The script will train the VGG-16 model on the training dataset, evaluate its performance on the validation dataset, and save the trained model for future use.

## Project Structure

The project is organized as follows:

- `vgg16_image_classification.py`: The main script to train and evaluate the VGG-16 model.
- `data/`: Directory to store the training, validation, and test datasets.
- `output/`: Directory to save the trained model.

## Code Structure

The code follows the following structure:

- Data Loading and Transformation:
  - `get_data(file_dir)`: Loads and transforms the data using PyTorch's `ImageFolder` and `DataLoader`.
- Model Creation and Modification:
  - `get_vgg16_pretrained_model(model_dir='', weights=models.vgg16_bn(pretrained=True).state_dict(), len_target=1000)`: Retrieves the VGG-16 pre-trained model and modifies its classifier for the desired number of output classes.
- Evaluation:
  - `eval_model(vgg, criterion, dataset='val')`: Evaluates the model's performance on the specified dataset.
  - `get_epoch_progress_graph(accuracy_train, loss_train, accuracy_val, loss_val, save_dir=out_plot_dir)`: Plots the progress of accuracy and loss during training epochs.
  - `get_classification_report(truth_values, pred_values)`: Generate a classification report and confusion matrix for the model predictions
  - `save_classification_report(truth_values, pred_values, out_report_dir)`: Save the report at a preset directory
- Training:
  - `train_model(vgg, criterion, optimizer, scheduler, num_epochs=10)`: Trains the model using the training dataset and evaluates its performance on the validation dataset.

## Results

- The trained VGG-16 model will be saved in the `output` directory as `VGG16_trained.pth`. You can use this model for inference on new images or further fine-tuning if needed.
- There will also be a plotted chart of all the epoch stats saved in `output` as `epoch_progress.jpg`
- Finally, a full classification report of the model when testing the `test` file will also be saved in `output` as `classification_report.txt`

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The VGG-16 model implementation is based on the torchvision library in PyTorch.
- The dataset loading and transformation code is adapted from PyTorch's official documentation.