# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# TODO: write script that takes 10% of training and splits it to validation
# import sklearn.model_selection
# sklearn.model_selection.model_selection


import os
import random
import shutil


# 10% = 968
# 5% = 484

# add 484 images

# images = []
# new_images = []
# cnt = 0
# for file in os.listdir("C:/Users/Sabin/Documents/SDC/RDW_Data/zips/rechte stukken 18-11-2021 14-59-21"):
#     images.append(file)

# random.shuffle(images)
# random.shuffle(images)
# random.shuffle(images)
# random.shuffle(images)

# for image in images:
#     cnt = cnt + 1
#     if cnt <= 484:
#         new_images.append(image)
        

# # print(len(new_images))
# # print(new_images)

# count =0 
# for image in new_images:
#     count = count +1

#     shutil.copy(f"C:/Users/Sabin/Documents/SDC/RDW_Data/zips/rechte stukken 18-11-2021 14-59-21/{image}", 
#             f"C:/Users/Sabin/Documents/SDC/SL_data/dataset_95p_turns/5p_turns/{image}")
            
# print("count", count)

# ------------------------------------------------------------------------------------------------
# remove 484 images

images = []
new_images = []
cnt = 0
for file in os.listdir("C:/Users/Sabin/Documents/SDC/SL_data/dataset_95p_turns/training/images 18-11-2021 14-59-21"):
    images.append(file)

random.shuffle(images)
random.shuffle(images)
random.shuffle(images)
random.shuffle(images)

for image in images:
    cnt = cnt + 1
    if cnt <= 484:
        new_images.append(image)
        

print(len(new_images))
# print(new_images)

for image in new_images:
    pass
    # print(image)
    os.remove(f"C:/Users/Sabin/Documents/SDC/SL_data/dataset_95p_turns/training/images 18-11-2021 14-59-21/{image}")
