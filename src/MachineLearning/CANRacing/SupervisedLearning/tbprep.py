import numpy as np
from PIL import Image, ImageDraw

def draw_pred_and_traget_npy(img: np.ndarray, filename: str, predicted_actions, target_actions, dataformats="HWC"):
    """Draws the image name in red, the predicted actions in blue, and the target actions in green in the top left corner of the image.
    This function only accepts numpy arrays.

    :param img The image as an npy array
    :param filename String of the filename
    """

    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)

    img = Image.fromarray(np.uint8(img), mode="RGB") # ImageDraw doesnt accept .npy so convert to Image

    draw = ImageDraw.Draw(img)
    draw.text((10, 10), str(f"Filename: {filename}"), fill=(255, 0, 0))
    draw.text((10, 20), str(f"Predicted steering: {predicted_actions[0][0]}"), fill=(0, 0, 255))
    draw.text((10, 30), str(f"Target steering: {target_actions[0][0]}"), fill=(0, 255, 0))
    draw.text((10, 40), str(f"Predicted throttle: {predicted_actions[0][1]}"), fill=(0, 0, 255))
    draw.text((10, 50), str(f"Target throttle: {target_actions[0][1]}"), fill=(0, 255, 0))
    img_with_preds = np.asarray(img, dtype="uint8") # convert back to numpy array

    return img_with_preds
