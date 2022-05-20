import os
import random
import cv2
import shutil

validationChance = 40
testingChance = 60

images = []
validation = []
testing = []

# empty directory with validation images
for filename in os.listdir("D:\\SDC Data\\validate\\"):
    file_path = os.path.join("D:\\SDC Data\\validate\\", filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# empty directory with testing images
for filename in os.listdir("D:\\SDC Data\\testing\\"):
    file_path = os.path.join("D:\\SDC Data\\testing\\", filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# get all image names
for image in os.listdir("D:\SDC Data\images 30-03-2022 15-17-40"):
    images.append(image)

imagesLength = len(images)
randomList = random.sample(range(imagesLength-1), round(validationChance / 100 * imagesLength))

# add image to validation based on random index
for index in randomList:
    validation.append(images[index])

# write image to validate directory and remove it from images
for image in validation:
    validateImage = cv2.imread("D:\\SDC Data\\images 30-03-2022 15-17-40\\"+image)
    cv2.imwrite("D:\\SDC Data\\validate\\"+image, validateImage)
    images.remove(image)

# the remaining images are the testing images
testing = images

# write every testing image to the testing directory
for image in testing:
    testingImage = cv2.imread("D:\\SDC Data\\images 30-03-2022 15-17-40\\"+image)
    cv2.imwrite("D:\\SDC Data\\testing\\"+image, testingImage)

print("Done")
