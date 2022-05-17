from unittest import result
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
import os
from tensorboard import program
from signal import signal

def launch_tb(log_dir, wait=True):
    """
    Automatically spins up a tensorboard at localhost:600x (Default 6006, increases by 1 if in use)

    :param log_dir The directory to store the tensorboard logs in
    """
    tb = program.TensorBoard()

    tb.configure(argv=[None, '--logdir', log_dir])

    url = tb.launch()
    print(f"Tensorboard {log_dir[-24:]} is available at: {url}")

    if wait:
        # signal.pause() # TODO: find windows version, i suppose it works without though
        pass

def create_tb(log_dir, wait=True):
    """
    Create and spin up a new tensorboard and remove any existing ones with the same log_dir

    :param log_dir The directory to store the tensorboard logs in
    :param wait Boolean if the program should continue to run after finishing to keep the tensorboard alive for examination
    """
    # remove if already exists
    # shutil.rmtree(os.path.abspath(log_dir), ignore_errors=True)

    writer = SummaryWriter(log_dir=log_dir)

    launch_tb(log_dir, wait=wait)
    
    return writer

def tb_show_text(text, epoch=0, name: str = "text", writer: SummaryWriter = None):
    """
    Send text to tensorboard

    :param text A string with text
    :param epoch The current epoch (Optional)
    :param name Name of the text to be sent to tensorboard
    :param writer The writer to be written to
    """
    if writer is None:
        print("No writer")

    writer.add_text(tag=name, text_string=str(text), global_step=epoch)
    writer.flush()

    return

def tb_show_loss(loss, epoch, name: str = "loss", writer: SummaryWriter = None):
    """
    Send the loss to tensorboard

    :param loss The loss
    :param epoch The current epoch
    :param name Name of the scalar to be sent to tensorboard
    :param writer  The writer to be written to
    """
    if writer is None:
        print("No writer")

    print("Loss:", loss)
    writer.add_scalar(name, loss, epoch)
    writer.flush()

    return

def tb_show_image(img: np.ndarray, epoch, name: str = "images", dataformats="CHW", writer=None):
    """
    Send an image to tensorboard

    :param img The image to be shown in tensorboard
    :param epoch The current epoch
    :param name The name of the heading
    :param dataformats The shape of the image, CHW or HWC
    :param writer The writer to be written to
    """
    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)
        dataformats = "HWC"

    if writer is None:
        print("No writer")
    
    writer.add_image(name, img_tensor=img, global_step=epoch, dataformats=dataformats)
    writer.flush()
