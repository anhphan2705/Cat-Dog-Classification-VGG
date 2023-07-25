# VGG-16 Image Classification Cat Dog and API

This repository contains Python scripts for training and evaluating an image classification model based on the VGG-16 architecture using PyTorch. The trained model is capable of classifying images into two categories: dogs and cats. Additionally, there is an API script that implements the trained model and allows users to classify multiple images as either dogs or cats.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Structure](#code-structure)
- [Results](#results)
- [API Usage](#api-usage)
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
- FastAPI
- Uvicorn

## Getting Started

1. Clone the repository:

  ```bash
  git clone https://github.com/anhphan2705/Image-Classification-Dog-Cat.git
  ```

2. Install the required dependencies:

  ```bash
  pip install torch torchvision matplotlib scikit-learn fastapi uvicorn
  ```

3. Download pre-trained model (optional):

- Here is a link to my trained model with a classification report available to download. It reported a 99.6% accuracy for my test file.

  ```link
  https://www.dropbox.com/s/nxllvz36o241dal/VGG-Train-9960.zip?dl=0
  ```

4. Prepare the Data:

- Place your training, validation, and test datasets in separate directories (`train`, `val`, `test`) inside the `data` directory as shown below:

  ```
  Image-Classification-VGG-Dog-Cat/
    cat_dog_train_model.py
    cat-dog-classifier-api.py
    pre-trained-model.pth (optional)
    output/
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

- If you want to load a pre-trained model from your local computer, add the `model_dir` argument to the `get_vgg16_pretrained_model()` function in the `cat_dog_train_model.py` script.

5. Training and Evaluation:

- Open the `cat_dog_train_model.py` file and modify the necessary parameters such as file directories, output directory, and your data files name.

  ```python
  file_dir = './data'
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

- The script will train the VGG-16 model on the training dataset, evaluate its performance on the validation dataset, and save the trained model for future use.

## Project Structure

The project is organized as follows:

- `cat_dog_train_model.py`: The main script to train and evaluate the VGG-16 model.
- `cat-dog-classifier-api.py`: The script for implementing the trained model as an API using FastAPI and Uvicorn.
- `data/`: Directory to store the training, validation, and test datasets.
- `output/`: Directory to save the trained model and evaluation results.

## Code Structure

### `cat_dog_train_model.py`

The code follows the following structure:

- Data Loading and Transformation:
  - `get_data(file_dir)`: Loads and transforms the data using PyTorch's `ImageFolder` and `DataLoader`.
- Model Creation and Modification:
  - `get_vgg16_pretrained_model(model_dir='', weights=models.vgg16_bn(pretrained=True).state_dict(), len_target=1000)`: Retrieves the VGG-16 pre-trained model and modifies its classifier for the desired number of output classes.
- Evaluation:
  - `eval_model(vgg, criterion, dataset='val')`: Evaluates the model's performance on the specified dataset.
  - `get_epoch_progress_graph(accuracy_train, loss_train, accuracy_val, loss_val, save_dir=out_plot_dir)`: Plots the progress of accuracy and loss during training epochs.
  - `get_classification_report(truth_values, pred_values)`: Generate a classification report and confusion matrix for the model predictions.
  - `save_classification_report(truth_values, pred_values, out_report_dir)`: Save the report at a preset directory.
- Training:
  - `train_model(vgg, criterion, optimizer, scheduler, num_epochs=10)`: Trains the model using the training dataset and evaluates its performance on the validation dataset.

### `cat-dog-classifier-api.py`

The code follows the following structure:

- API Routes:
  - `/`: Serves the root route and displays a welcome message with a link to the API documentation.
  - `/dog-cat-classification`: API endpoint to classify multiple images as either dogs or cats using the fine-tuned VGG-16 model.
- Image Processing Functions:
  - `convert_byte_to_arr(byte_image)`: Convert an image in byte format to a PIL Image object (RGB format).
  - `convert_arr_to_byte(arr_image)`: Convert a numpy array image (RGB format) to byte format (JPEG).
  - `multiple_to_one(images)`: Combine multiple images horizontally into a single image.
  - `assign_image_label(images, labels, font="arial.ttf", font_size=25)`: Add labels to the input images.
  - `get_data(np_images)`: Prepare the list of numpy array images for classification.
  - `get_vgg16_pretrained_model(model_dir=MODEL_DIRECTORY, weights=models.VGG16_BN_Weights.DEFAULT)`: Retrieve the VGG-16 pre-trained model and modify the classifier with a fine-tuned one.
  - `get_prediction(model, images)`: Perform image classification using the provided model.
- API Endpoints:
  - `welcome_page()`: Serves the root route ("/") and displays a welcome message with a link to the API documentation.
  - `dog_cat_classification(in_images: list[UploadFile])`: API endpoint to classify multiple images as either dogs or cats using a fine-tuned VGG-16 model.

## Results

- The trained VGG-16 model will be saved in the `output` directory as `VGG16_trained.pth`. You can use this model for inference on new images or further fine-tuning if needed.
- There will also be a plotted chart of all the epoch stats saved in `output` as `epoch_progress.jpg`.
- Finally, a full classification report of the model when testing the `test` file will also be saved in `output` as `classification_report.txt`.

## API Usage

1. Create a classification model:

- You can train a model by using the `cat_dog_train_model.py` script. See [Getting Started](#getting-started) step 4 and 5.
- Or you can download my model that I trained with accuracy of 99.6% [here](https://www.dropbox.com/s/nxllvz36o241dal/VGG-Train-9960.zip?dl=0)

2. Follow the suggested folder directory

- Place the model somewhere, preferably as shown in [Getting Started](#getting-started) step 4.
- Modify the `MODEL_DIRECTORY` constant in the `cat-dog-classifier-api.py` script accordingly at line 15

  ```python
   MODEL_DIRECTORY = './output/VGG16_trained_9960.pth'
  ```

3. Adjust variables:

- You can also adjust some varibles as you prefer for the API response at line 16 and 17. These font setting are for the labels that will be print on the output images

  ```python
   FONT = "arial.ttf"
   FONT_SIZE = 25
  ```

4. Run the API:

```bash
uvicorn cat-dog-classifier-api:app --host 0.0.0.0 --port 8000
```

5. Access the API documentation:

Go to `http://localhost:8000/docs` in your web browser to access the API documentation and interact with the `/dog-cat-classification` endpoint.

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The VGG-16 model implementation is based on the torchvision library in PyTorch.
- The dataset loading and transformation code is adapted from PyTorch's official documentation.
- The API implementation is based on FastAPI and Uvicorn.