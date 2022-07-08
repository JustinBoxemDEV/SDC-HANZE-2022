"""
This script is used for splitting the dataset in training and testing.

Source:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""
import os
import shutil
from sklearn.model_selection import train_test_split


# insert folder path here (a folder within bochten)
images = os.listdir(f'D:/KWALI/images 07-06-2022 16-37-30')

# split 10% (this method is meant for splitting train/test but we use it to extract 10%)
train, test = train_test_split(images, test_size=0.10, random_state=42)

print("train has a size of", len(train))
print("validation has a size of", len(test))

# Move the images to 'training/testing' folder
# (i am moving instead of copying to check if all images are processed, make sure you have a backup dataset if you do this. Otherwise use shutil.copy)
for img in train:
    print(f'Moving {img} to training set')
    # img
    shutil.copy(
        f'D:/KWALI/images 07-06-2022 16-37-30/{img}',
        f'D:/KWALI/training/{img}')

for img in test:
    print(f'Moving {img} to testing set')
    shutil.copy(
        f'D:/KWALI/images 07-06-2022 16-37-30/{img}',
        f'D:/KWALI/validation/{img}')
