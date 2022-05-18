# Used for GPU acceleration
# Check tensorflow info (see if it detects your GPU)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import tensorflow as tf
import torch as th

# print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

# print("GPU available: ", tf.test.is_gpu_available())


print("GPU available: ", th.cuda.is_available())