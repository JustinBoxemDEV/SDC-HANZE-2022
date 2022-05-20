from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tensorboard import program
import numpy as np
from PIL import Image, ImageDraw
import shutil
import os

def launch_tb(log_dir: str, wait=True):
    """
    Automatically spins up a tensorboard at localhost:600x (Default 6006, increases by 1 if in use)

    :param log_dir The directory to store the tensorboard logs in
    """
    tb = program.TensorBoard()

    tb.configure(argv=[None, '--logdir', log_dir])

    url = tb.launch()
    print(f"Tensorboard {log_dir[-24:]} is available at: {url}")

    if wait:
        # signal.pause() # TODO: find windows version, i believe it works without though
        pass

def create_tb(log_dir, wait=True):
    """
    Create and spin up a new tensorboard and remove any existing ones with the same log_dir

    :param log_dir The directory to store the tensorboard logs in
    :param wait Boolean if the program should continue to run after finishing to keep the tensorboard alive for examination
    """
    # remove if already exists
    shutil.rmtree(os.path.abspath(log_dir), ignore_errors=True)

    writer = SummaryWriter(log_dir=log_dir)

    launch_tb(log_dir, wait=wait)
    
    return writer

def tb_show_text(text: str, epoch: int = 0, name: str = "text", writer: SummaryWriter = None):
    """
    Send text to tensorboard

    :param text A string with text
    :param epoch The current epoch (Optional)
    :param name Name of the text header to be sent to tensorboard
    :param writer The writer that writes to your current training tensorboard
    """
    if writer is None:
        print("No writer")
    else:
        writer.add_text(tag=name, text_string=str(text), global_step=epoch)
        writer.flush()

    return

def tb_show_loss(loss, epoch: int, name: str = "loss", writer: SummaryWriter = None):
    """
    Send the loss to tensorboard

    :param loss The loss
    :param epoch The current epoch
    :param name Name of the scalar to be sent to tensorboard
    :param writer  The writer that writes to your current training tensorboard
    """
    if writer is None:
        print("No writer")
    else:
        # print(f"Loss: {loss} at epoch {epoch}")
        writer.add_scalar(name, loss, epoch)
        writer.flush()

    return

def tb_show_image(img: np.ndarray, epoch: int, name: str = "images", dataformats="CHW", writer: SummaryWriter = None):
    """
    Send an image to tensorboard

    :param img The image to be shown in tensorboard as nparray
    :param epoch The current epoch
    :param name The name of the heading
    :param dataformats The shape of the image, CHW or HWC
    :param writer The writer that writes to your current training tensorboard
    """
    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)
        dataformats = "HWC"

    if writer is None:
        print("No writer")
    
    writer.add_image(name, img_tensor=img, global_step=epoch, dataformats=dataformats)
    writer.flush()

def draw_pred_and_target_npy(img: np.ndarray, filename: str, predicted_actions, target_actions, dataformats="HWC"):
    """
    Draws the image name in red, the predicted actions in cyan, and the target actions in green in the top left corner of the image.
    This function only accepts numpy arrays.

    :param img The image as an npy array
    :param filename String of the filename
    :param predicted_actions The predicted actions by the model
    :param target_actions The ground truth actions
    :param dataformats HWC or CHW
    """
    if dataformats == "CHW":
        # print("changing dataformat to HWC")
        img = np.moveaxis(img, 0, 2)
    
    img = Image.fromarray(np.uint8(img), mode="RGB") # ImageDraw doesnt accept .npy so convert to Image
    
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), str(f"Filename: {filename}"), fill=(255, 0, 0))
    draw.text((10, 20), str(f"Predicted steering: {predicted_actions[0][0]}"), fill=(0, 255, 255))
    draw.text((10, 30), str(f"Target steering: {target_actions[0][0]}"), fill=(0, 255, 0))
    draw.text((10, 40), str(f"Predicted throttle: {predicted_actions[0][1]}"), fill=(0, 255, 255))
    draw.text((10, 50), str(f"Target throttle: {target_actions[0][1]}"), fill=(0, 255, 0))
    img_with_preds = np.asarray(img, dtype="uint8") # convert back to numpy array

    return img_with_preds
