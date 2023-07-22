from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from torchvision import transforms
from skimage.transform import resize


app = FastAPI()


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
    preprocess = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    data = preprocess(np_images)
    data = data.unsqueeze(0)
    return data
    

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

    data = get_data(images)

    byte_stitched_image = convert_arr_to_byte(images)
    return Response(byte_stitched_image, media_type="image/jpg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
