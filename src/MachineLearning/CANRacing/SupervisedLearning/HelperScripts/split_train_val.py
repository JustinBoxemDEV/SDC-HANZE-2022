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
    shutil.move(
        f'11-45-20',
        f'90p_11-45-20')

# we dont need the remaining 10% so we ignore
# for img in test:
#     print(f'Moving {img} to testing set')
#     shutil.move(
#         f'11-45-20',
#         f'10p_11-45-20')
