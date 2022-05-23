# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

"""
This script is used for splitting the dataset in training and testing.

Warning: Do not run this script blindly, it moves images :)
"""
import os
import shutil
from sklearn.model_selection import train_test_split


# insert folder path here (a folder within bochten)
images = os.listdir(f'11-45-20')

# split 10% (this method is meant for splitting train/test but we use it to extract 10%)
train, test = train_test_split(images, test_size=0.10, random_state=42)

print("train has a size of", len(train))
print("test has a size of", len(test))

# Move the images to 'training/testing' folder
# (i am moving instead of copying to check if all images are processed,
# make sure you have a backup dataset if you do this. Otherwise use shutil.copy)
for img in train:
    print(f'Moving {img} to training set')
    # img
    shutil.copy(
        f'11-45-20',
        f'90p_11-45-20')

# we dont need the remaining 10% so we ignore it
# for img in test:
#     print(f'Moving {img} to testing set')
#     shutil.move(
#         f'11-45-20',
#         f'10p_11-45-20')


# -----------------------------------------------------------------------
# shitty script
import os
import random
import shutil

# split  5880 images

images = []
val_images = []
test_images = []
cnt = 0
for file in os.listdir("D:/SDC/sdc_data/justin_data/validation/40percent_val_images 30-03-2022 15-17-40"):
    images.append(file)

random.shuffle(images)
random.shuffle(images)
random.shuffle(images)
random.shuffle(images)

for image in images:
    cnt = cnt + 1
    if cnt <= 2352:
        val_images.append(image)
    else:
        test_images.append(image)
        

print(len(val_images))
print(len(test_images))
# print(new_images)

for image in val_images:
    pass
    # print(image)
    shutil.copy(f"D:/SDC/sdc_data/justin_data/validation/40percent_val_images 30-03-2022 15-17-40/{image}", f"D:/SDC/sdc_data/justin_data/validation/val_images 30-03-2022 15-17-40/{image}")

for image in test_images:
    pass
    # print(image)
    shutil.copy(f"D:/SDC/sdc_data/justin_data/validation/40percent_val_images 30-03-2022 15-17-40/{image}", f"D:/SDC/sdc_data/justin_data/testing/test_images 30-03-2022 15-17-40/{image}")
