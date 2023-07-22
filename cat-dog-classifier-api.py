from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from skimage.transform import resize
import time


app = FastAPI()
MODEL_DIRECTORY = './output/VGG16_trained_9960.pth'


def convert_byte_to_arr(byte_image):
    """
    Converts a byte image to a NumPy array.

    Args:
        byte_image (bytes): The byte representation of the image.

    Returns:
        numpy.ndarray: The NumPy array representation of the image.
    """
    arr_image = np.array(Image.open(BytesIO(byte_image)))
    return arr_image


def convert_arr_to_byte(arr_image):
    """
    Converts a NumPy array to a byte image.

    Args:
        arr_image (numpy.ndarray): The NumPy array representation of the image.

    Returns:
        bytes: The byte representation of the image.
    """
    arr_image_cvt = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    success, byte_image = cv2.imencode(".jpg", arr_image_cvt)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("Cannot convert array image to byte image")


def get_data(np_images):
    data_transform = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Load data into dataloaders
    # dataloaders = torch.utils.data.DataLoader(
    #         datasets_img,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=1
    #     )

    data = data_transform(np_images)
    data = data.unsqueeze(0)
    return data

def get_vgg16_pretrained_model(model_dir = MODEL_DIRECTORY , weights=models.VGG16_BN_Weights.DEFAULT):
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
    vgg16.load_state_dict(torch.load(model_dir))
    vgg16.eval()
    print("[INFO] Loaded VGG-16 pre-trained model\n", vgg16, "\n")
    
    return vgg16


@app.get("/")
def welcome_page():
    """
    Serves the root route ("/") and displays a welcome message with a link to the API documentation.
    """
    return HTMLResponse(
        """
        <h1>Welcome to Banana</h1>
        <p>Click the button below to go to /docs/:</p>
        <form action="/docs" method="get">
            <button type="submit">Visit Website</button>
        </form>
    """
    )


@app.post("/dog-cat-classification")
async def dog_cat_classification(in_images: list[UploadFile]):
    """
    API endpoint to stitch multiple images together.

    Args:
        in_images (List[UploadFile]): List of images to be stitched.

    Returns:
        Response: The stitched image as a response.

    Raises:
        Exception: If stitching fails or the image cannot be cropped.
    """
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
    # Preping data n model
    data = get_data(images)
    vgg = get_vgg16_pretrained_model()
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    if use_gpu:
        torch.cuda.empty_cache()
        vgg.cuda()

    byte_image = convert_arr_to_byte(images)
    return Response(byte_image, media_type="image/jpg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
