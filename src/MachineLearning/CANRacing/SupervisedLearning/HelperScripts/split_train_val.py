# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# TODO: write script that takes 10% of training and splits it to validation
# import sklearn.model_selection
# sklearn.model_selection.model_selection

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
