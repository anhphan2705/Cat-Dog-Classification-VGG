# VGG-16 Image Classification with PyTorch

This repository contains code for training and evaluating an image classification model based on the VGG-16 architecture using PyTorch. It will be able to tell if the input image is a dog or a cat after training.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- Matplotlib (optional)

## Getting Started

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/Image-Classification-Dog-Cat.git
   ```

2. Install the required dependencies:

   ```shell
   pip install torch torchvision matplotlib
   ```

3. Prepare the Data:

   - Place your training, validation, and test datasets in separate directories (`train`, `val`, `test`) inside the `data` directory.

4. Training and Evaluation:

   - Open the `cat_dog_train_model.py` file and modify the necessary parameters such as file directories, hyperparameters, and number of classes.

   - Run the script:

     ```shell
     python ./cat_dog_train_model.py
     ```

   - The script will train the VGG-16 model on the training dataset, evaluate its performance on the validation dataset, and save the trained model for future use.

## Project Structure

The project is organized as follows:

- `vgg16_image_classification.py`: The main script to train and evaluate the VGG-16 model.
- `data/`: Directory to store the training, validation, and test datasets.
- `output/`: Directory to save the trained model.

## Code Structure

The code follows the following structure:

- Data Loading and Transformation:
  - `get_data(file_dir, TRAIN='train', VAL='val', TEST='test')`: Loads and transforms the data using PyTorch's `ImageFolder` and `DataLoader`.
- Model Creation and Modification:
  - `get_vgg16_pretrained_model(model_dir='', weights=models.vgg16_bn(pretrained=True).state_dict(), len_target=1000)`: Retrieves the VGG-16 pre-trained model and modifies its classifier for the desired number of output classes.
- Evaluation:
  - `eval_model(vgg, criterion, dataset='val')`: Evaluates the model's performance on the specified dataset.
- Training:
  - `train_model(vgg, criterion, optimizer, scheduler, num_epochs=10)`: Trains the model using the training dataset and evaluates its performance on the validation dataset.

## Results

The trained VGG-16 model will be saved in the `output` directory as `VGG16_trained.pth`. You can use this model for inference on new images or further fine-tuning if needed.

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The VGG-16 model implementation is based on the torchvision library in PyTorch.
- The dataset loading and transformation code is adapted from PyTorch's official documentation.