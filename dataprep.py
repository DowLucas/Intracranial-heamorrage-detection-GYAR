import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os
from matplotlib import style
from pydicom.data import get_testdata_files
import matplotlib as mpl
from zipfile import ZipFile
import pickle

style.use("seaborn-pastel")
SUB_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

Extract = True

train_images = []

n = 0
'''for index, row in df.iterrows():
    id = str(row["ID"]).split("_", 2)[:2]
    id = id[0]+"_"+id[1]
    train_images.append(id)
    n += 1

    if n > 20000:
        break'''


EXTRACT_NUM_IMAGES = 50000
EXTRACTED_IMAGES_PATH = "stage_1_train_images"

try:
    images_already_extacted = os.listdir(EXTRACTED_IMAGES_PATH)
except:
    pass

def getTrainImages(wholeList):
    returnList = []
    for w in wholeList:
        if "stage_1_train_images" in w:
            returnList.append(w)

    return returnList

data = pickle.load(open("IDdict.pickle", "rb"))
print(data)
PostitiveCheck = False

imagesExtracted = []
LabelsArray = np.zeros((50000, 1))

num = 0
zipPath = "stage_1_train_images/"
if Extract:
    print("Starting Extraction of images...")
    with ZipFile("rsna-intracranial-hemorrhage-detection.zip", "r") as zip:
        images = getTrainImages(zip.namelist())
        print("Loaded {} Image Names".format(len(images)))
        for ID in data.keys():
            image_path = zipPath+ID+".dcm"
            if data[ID][-1] == 1 and PostitiveCheck:
                PostitiveCheck = not PostitiveCheck
                zip.extract(image_path, path="imageData")
                LabelsArray[num] = data[ID][-1]
                num += 1
                imagesExtracted.append(ID)

            elif data[ID][-1] == 0 and not PostitiveCheck:
                PostitiveCheck = not PostitiveCheck
                zip.extract(image_path, path="imageData")
                LabelsArray[num] = data[ID][-1]
                num += 1
                imagesExtracted.append(ID)
            else:
                continue

            if num%100 == 0:
                print(f"{num} Images extracted")
            if num == 50000:
                break




print("Complete with extraction!")
pickle.dump(imagesExtracted, open("extractedImageId.pickle", "wb"))
pickle.dump(LabelsArray, open("train/labels.pickle", "wb"))


def translate(value, leftMin, leftMax, rightMin, rightMax):
    #print(value, leftMin, leftMax, rightMin, rightMax)
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


import pydicom as dicom
from skimage.transform import resize
PNG = False


# make it True if you want in PNG format

# Specify the .dcm folder path
folder_path = "ImageData/stage_1_train_images"
# Specify the output jpg/png folder path
jpg_folder_path = "ImageData/JPG_test"
images_path = os.listdir(folder_path)

# Variables
IMG_PX_SIZE = 100
IMAGE_PIXEL_INTENSIFIER = 5000
NUM_IMAGES = len(images_path)
image_pixel_data = np.zeros((NUM_IMAGES, IMG_PX_SIZE, IMG_PX_SIZE))

print(image_pixel_data.shape)
IMAGE_NAME = []

data = {}

for N, image in enumerate(imagesExtracted):
    ds = dicom.dcmread(folder_path+"/"+image+".dcm")
    resized_img = resize(ds.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))

    #image_pixel_data[N] = resized_img
    IMAGE_NAME.append(image)
    data[image] = [resized_img, LabelsArray[N]]

    if (N+1)%50 == 0:
        print("Completed {} -> {} %".format(N, 100*N/len(images_path)))


print("Complete... Packaging")

pickle.dump(data, open("train/all_data.pickle", "wb"))
pickle.dump(IMAGE_NAME, open("train/image_names.pickle", "wb"))