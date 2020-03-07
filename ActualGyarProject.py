'''
Code Written by Lucas Dow, NV3E.
Gymnasiearbete @ IEGS
Method for classifying intracranial hemorrhage in paitents CT-Scans, step by step guide, explanation of each step is provided.



'''Import modules which makes calculations easier'''
'''Numpy is a library which helps with array/list/tensor manipulation such as resizing arrays and data'''
import numpy as np

'''Library which can create/read files containing any type of data'''
import pickle

'''Windows OS library which helps with locating files in certain directories'''
import os

'''Helps to create dataframes and tables'''
import pandas as pd

'''Used to time how long different processes take'''
import time

'''Random for doing random operations'''
import random

'''For showing images'''
import matplotlib.pyplot as plt

import cv2

'''The data is found inside a ZIP archive. This file is over 150 GB in size so it would
not be a good idea to extract all the images directly onto a hard drive, Therefore, a library called
zipfile is used to extract the images'''
from zipfile import ZipFile

'''Able to resize pixel arrays'''
from skimage.transform import resize

'''The labels (the data the defines if and which type of intracranial hemorrhage is present) is
stored in a csv file. By using pandas build in read_csv function, the file can be read and stored in a table/dataframe
the dataframe will receive the variable "df", we can also define "csvFile" and set it equal to the name of the file'''

# Defining csv file path
csvFile = "stage_1_train.csv"
df = pd.read_csv(csvFile) if os.path.exists(csvFile) else None

print(df.head())

'''
In order to see how the table looks, df.head() is run to show the first 5 rows of the dataframe.
  ID                               Label
0          ID_63eb1e259_epidural      0
1  ID_63eb1e259_intraparenchymal      0
2  ID_63eb1e259_intraventricular      0
3      ID_63eb1e259_subarachnoid      0
4          ID_63eb1e259_subdural      0'''

'''Show the length of the dataframe'''
print(len(df))
'''Length: 4045572 rows'''

'''Each ID in the dataframe has 6 rows for each of the sub types of intraccranial hemorrhage, meaning that the there are 674262 unique
IDs in the dataframe, example ID_1 has 6 rows -> ID_1_subtype1, ID_1_subtype2 ... ID_1_subtype6'''

'''The goal is to create a dictionary where the input is unique ID (without the subtype) and the definition/output is a scalar value which is
 either 0, 1 depending on if any the patent is suffering from any subtype of intracranial hemorrhage. For instance, if a paitent with a
 unique ID is suffering from epidural intracranial hemorrhage, the output/definition will be a 1 and if the patent is not suffering from
 any subtype, the value will be 0'''

'''First of all, a new column in the dataframe variable "df" has to be created where the subtype in the ID removed for instance ID_1_subtype1
would be changed to ID_1'''

df["OnlyID"] = df["ID"].apply(lambda x: x.replace("_epidural", "").replace("_intraparenchymal", "").replace("_intraventricular", "").replace(
        "_subarachnoid", "").replace("_subdural", "").replace("_any", ""))

'''Looking at the first 5 rows of the dataframe with the new column'''
print(df.head())
'''   ID                            Label   OnlyID
0          ID_63eb1e259_epidural      0  ID_63eb1e259
1  ID_63eb1e259_intraparenchymal      0  ID_63eb1e259
2  ID_63eb1e259_intraventricular      0  ID_63eb1e259
3      ID_63eb1e259_subarachnoid      0  ID_63eb1e259
4          ID_63eb1e259_subdural      0  ID_63eb1e259'''

'''Now, lets create an empty dictionary and give it the variable "data"'''
data = {}

'''A dictionary might look like this {"Subject1": 2, "Subject2": 5, "Subject3": 10}
The basic structure is {"Key": definition}'''

# Iterate over all rows in the dataframe
# "row" is the variable that each row is stored in
for row in df.iterrows():
    # If this program has already been run before and the file already exists (read about this later),
    # then break from the loop and skip this step
    if os.path.isfile("IDandLabels.pickle"):
        # If the file exists, set the data variable equal to the previously created dictionary
        data = pickle.load(open("IDandLabels.pickle", "rb"))
        # Break from for-loop
        break

    # Prints only the first row of the dataframe
    print(row) if row[0] == 0 else None
    '''Lets take a look at what the variable "row" stores

        (0, ID        ID_63eb1e259_epidural
        Label                         0
        OnlyID             ID_63eb1e259
        Name: 0, dtype: object)

    Looks like the row object stores a so called python tuple where the first element indicates the row index
    within the dataframe and second element is a dictionary of length 3, where each entry represents each column
    in the dataframe, ID, Label and OnlyID'''

    # Defining a variable "ID" for each row by setting "ID" equal to the OnlyID column of each row
    # By using row[1], the second element in the variable "row" can be accessed which was a dictionary
    # The OnlyID can then be accessed by using ["OnlyID"]
    ID = row[1]["OnlyID"]
    # Defining a variable "label" which is the Label column of the dataframe
    label = row[1]["Label"]

    # Printing the ID and label of the first row of the dataframe
    print(f"The first row of the dataframe has an ID of '{ID}' and a label of '{label}'") if row[0] == 0 else None

    '''
        Output:
        The first row of the dataframe has an ID of 'ID_63eb1e259' and a label of '0'
    '''

    '''Now that the ID and label of the row has been defined, a check has to be conducted to see if the
    ID or is not already an element/key of the data dictionary created earlier.
    This can be done be done using an if statement'''

    # data.keys() is a function which yields an array of all the keys in the data dictionary
    # A check is conducted to check if the ID is a key in the "data" dictionary
    if ID not in data.keys():
        # Everything which is indented will be executed if the above statement is true
        '''If the ID is not a key in the "data" dictionary, we can add it to the dictionary
        This can be done by the following line. This will add {ID: Label} to the dictionary'''
        data[ID] = label
    else:
        # This code will be executed if the ID IS a key in "data" dictionary
        '''If the ID is a key of the dictionary, we have to check if the current value/defentition of the value
        is 0 or 1 since this will determine if the value should be altered or not. If the value is 0, and the new value
        is 1, then we change the value to 1 since intracranial hemorrhage is present the patient's brain.
        If the value that already exists is 1 and the new value is 0, then no action is necessary because
        intracranial hemorrhageis already present as another subtype in the patient's brain'''
        # This can be done by the following
        # This is basically saying: Set the label of the data entry with ID -> ID equal to 1 if and only if the
        # current label in the dictionary is 0 and the new one is 1 else set it to 0.
        data[ID] = 1 if data[ID] == 0 and label == 1 else 0


# Checking the length of the "data" dictionary
print(len(data))
'''
    Length of data: 674258
    It looks like we lost 4 rows from the original 674262 but this not a minor loss so it will not looked into
'''

#print(data)
'''
    Lets have a look at 5 random entries in the dictionary: 
    {... 'ID_d5c6668fe': 0, 'ID_59f0d8bde': 1, 'ID_12d0a8d72': 0, 'ID_8f47f9abd': 0, 'ID_e1a08b43d': 0, ...}
'''

# Creates a file, which prevents the program to create a new "data" directory each time the program is executed.
pickle.dump(data, open("IDandLabels.pickle", "wb")) if not os.path.isfile("IDandLabels.pickle") else None

'''Lets check how many negative entries exists (label = 0) and positive (label = 1) this can be done by creating an 
array where the first element represents the amount of negative values and the second element represents the 
amount of positives values'''

labelDifferencesArray = np.zeros((2))

# Iterating over all keys in the "data" dictionary
for k in data.keys():
    # +1 to either the first or second element depending on the value of the label (0 or 1)
    labelDifferencesArray[data[k]] = labelDifferencesArray[data[k]] + 1

print(f"There are {labelDifferencesArray[0]} patients who tested negative and {labelDifferencesArray[1]} who tested positive for intracranial hemorrhage")

'''
    Output:
    There are 612088.0 patients who tested negative and 62170.0 who tested positive for intracranial hemorrhage
'''

'''The next step is to extract the images and read their pixel values, this would mean that we create a dictionary which
instead of only having the label as a definition would also contain an array of the pixel values in the picture. 
So in this case, the definition/value of each unique ID in the dictionary will be {ID: [PixelArray, Label]}
This will, however, not be done on all images since we dont have an equal distribution between patients with and 
without intracranial hemorrhage. Instead, since there is about 55000 patients diagnosed with some type of IH, there 
will a total 120 000 images that will be used since this will allow a 50 50 distribution between patients with and 
without IH. This leaves about 7000 images which the algorithm can be tested on to calculate accuracy for instance'''

# Lets start by defining some variables which will be required to extract the images

# Name of archive .ZIP file
zipfileName = "rsna-intracranial-hemorrhage-detection.zip"

# Location of new images
extractImagePath = "GYARimageData"

os.mkdir(extractImagePath) if not os.path.exists(extractImagePath) else None

# Variable for storing all the already extracted images
extracedImagesPath = "extractedImageId.pickle"
IDsOfExtractedImages = pickle.load(open(extracedImagesPath, "rb")) if os.path.isfile(extracedImagesPath) else []

# Number of images to be extracted for each label (positive and negative)
numberOfimagesToExtract = 55000

# Name of directory in ZIP archive
zipPath = "stage_1_train_images/"

numberOfImagesExtracted = 0

# Defining a function which is used to display all the names in the zip archive
def getTrainImages(wholeList):
    returnList = []
    for w in wholeList:
        if "stage_1_train_images" in w:
            returnList.append(w)

    return returnList

# The extraction should not occur everytime so a boolean varible is set and determinds if the images should be extracted or not
extract = False

'''Now that some varibles have been defined the image extraction process can begin'''

# Opeing the "zipfileName" which was defined above, "r" stands for read. The file is assigned the variable "zip"
if extract:
    with ZipFile(zipfileName, "r") as ZIP:
        # "allImages" is defined which will store all the file paths to the images within the zip archive
        allImages = getTrainImages(ZIP.namelist())
        print("Starting extraction of images where patients have NOT been diagnosed with IH")

        # Iterating over each key in the "data" dictionary
        # enumerate function sets the value "n" equal to the current index number of the data.keys() array, 1, 2, 3 and so on.
        for key in data.keys():
            # "image_path" if the full path to the image inside the .ZIP archive. The image is a DICOM image, hence the
            # .dcm extension (this is similar to JPEG or PNG)
            image_path = zipPath + key + ".dcm"

            # If the label (0 or 1) is equal to 0 and if it has not already been extracted -> extract it and add it to the
            # extracted images array
            if data[key] == 0:
                # Extracting image to previously defined image path
                ZIP.extract(image_path, path=extractImagePath) if not os.path.exists(extractImagePath+"/stage_1_train_images/"+key+".dcm") else None

                # Appending key to array
                IDsOfExtractedImages.append(key)

                # Adding 1 to number of images extracted
                numberOfImagesExtracted += 1

            # If the desired number of images have been extracted (in this case 55000) break the loop
            if numberOfImagesExtracted == numberOfimagesToExtract:
                break
            elif numberOfImagesExtracted%1000 == 0:
                # If the number of images extracted mod 1000 is equal to 0 then print the number of images extracted
                #print(f"{numberOfImagesExtracted} Images have been extracted...")
                pass

        print("Starting extraction of images where patients have been diagnosed with IH\n")
        numberOfImagesExtracted = 0

        # This part is similar to the one above, however, it extracts images with label of 1 instead of 0
        for key in data.keys():
            image_path = zipPath + key + ".dcm"
            # here data[key] == 1 instead of == 0 in the one above.
            if data[key] == 1:
                ZIP.extract(image_path, path=extractImagePath) if not os.path.exists(extractImagePath+"/stage_1_train_images/"+key+".dcm") else None
                IDsOfExtractedImages.append(key)
                numberOfImagesExtracted += 1
            if numberOfImagesExtracted == numberOfimagesToExtract:
                break
            elif numberOfImagesExtracted%1000 == 0:
                #print(f"{numberOfImagesExtracted} Images have been extracted...")
                pass
# Create a file with all the IDs of the extracted files only if it has not yet been created
pickle.dump(IDsOfExtractedImages, open("IDsOfExtractedImages.pickle", "wb")) if not os.path.exists("IDsOfExtractedImages.pickle") else None

'''Now that the images have been extracted, we can perform some checks to ensure that the correct amount of images have
been extracted and that 50% are negative IH and 50% are positive IH'''

# Gives us the number of images extracted | Expected amount: 110000
print(len(os.listdir(extractImagePath+"/stage_1_train_images")))
'''
    Output:
    110000 Images
'''

'''Now, lets check if half the images are labled with negative IH and the other half positive IH'''

# Lets define a new dictionary called "distribution" will will keep 2 values, positive & negative which will start at 0

distribution = {"positive": 0, "negative": 0}

newData = {}

# Iterate over all the extracted images

for image in os.listdir(extractImagePath+"/stage_1_train_images"):
    # Define the image name by removing the file extension .dcm
    # This replaces the extension .dcm with nothing, yielding only the image name, i.e the ID
    imageName = image.replace(".dcm", "")

    # Now lets get the label using the "data" dictionary
    label = data[imageName]

    # This will be our new data dictionary containing only the 110000 images and labels which were extracted
    newData[imageName] = data[imageName]

    # If the label is equal to 1 add one to the positive part of the "distribution" dictionary
    if label == 1:
        distribution["positive"] += 1
    # Else if the label is equal to 0 add one to the negative part of the "distribution" dictionary
    elif label == 0:
        distribution["negative"] += 1

# Now that this is complete, we can set the "data" variable equal to the "newData" variable and erase thevariablee from
# memory

data = newData
del newData

# Lets have a look at how many of the pictures extracted where negatively and politely labeled with IH.
print(f"There are {distribution['positive']} images which depict positive IH,\nand {distribution['negative']} images which depict negative IH")

'''
    Output:
    There are 55000 images which depict positive IH 
    and 55000 images which depict negative IH
'''

'''The extraction process worked! 55000 images of each label were successfully extracted. Now that this step has been
completed, the next step can commence. This step involves retrieving the pixel values of each image. Each image will be
resized to a 100x100 image which will result in 10000 total pixels in each image. The images are of type DICOM which
means that a library called pydicom will be used in order to extract the pixel values from each image.'''

'''Importing the pydicom library which is used to read the images'''
import pydicom

'''The math library is useful because it comes with many functions such as floor, cos, sin, power and so on'''
import math

# In order to free up memory on the computer, some variables which are no longer used are deleted
try:
    del distribution
    del allImages
    del df
except: pass

'''Lets start as before, by defining variables which will be used in this process'''

# Directory where all the training/evaluationg data will be stored
os.mkdir("GYARprocessedData") if not os.path.exists("GYARprocessedData") else None

# Pixel dimension
PIXEL_SIZE = 100

# The smaller dictionary which will only contain 5000 entries at a time
dataDivided = {}

# Creating a list/array of the keys within the "data" dictionary
keys = list(data.keys())

# Creating an evaluate images dictionary which will be used later to evaluate the images
# The last 5000 images of the data dictionary will be the evaluation images as seen in the coming section.
evaluate_images = {i: data[i] for i in keys[-5000:]}

numFiles = len(os.listdir("GYARprocessedData"))

# Next step is to iterate over all the keys in the "data" dictionary
for n, image in enumerate(keys):
    # Check if file already exists
    if int(math.floor(n/5000)) < numFiles:
        del data[image]
        continue


    # Defining the full image path to the image
    # The image varible does not include the .dcm extension so it needs to be added
    imagePath = extractImagePath+"/stage_1_train_images/"+image+".dcm"

    # Creating variable "ds" which is an object containing all the values of the image, including its pixel array
    try:
        ds = pydicom.dcmread(imagePath)
    except Exception as e:
        print(e)
        continue

    '''Lets have a look at an example of what the "ds" object looks like by printing it out'''
    print(ds) if n == 0 else None

    '''
        Output:
        (0008, 0018) SOP Instance UID                    UI: ID_000039fa0
        (0008, 0060) Modality                            CS: 'CT'
        (0010, 0020) Patient ID                          LO: 'ID_eeaf99e7'
        (0020, 000d) Study Instance UID                  UI: ID_134d398b61
        (0020, 000e) Series Instance UID                 UI: ID_5f8484c3e0
        (0020, 0010) Study ID                            SH: ''
        (0020, 0032) Image Position (Patient)            DS: ['-125.000000', '-141.318451', '62.720940']
        (0020, 0037) Image Orientation (Patient)         DS: ['1.000000', '0.000000', '0.000000', '0.000000', '0.968148', '-0.250380']
        (0028, 0002) Samples per Pixel                   US: 1
        (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
        (0028, 0010) Rows                                US: 512
        (0028, 0011) Columns                             US: 512
        (0028, 0030) Pixel Spacing                       DS: ['0.488281', '0.488281']
        (0028, 0100) Bits Allocated                      US: 16
        (0028, 0101) Bits Stored                         US: 16
        (0028, 0102) High Bit                            US: 15
        (0028, 0103) Pixel Representation                US: 1
        (0028, 1050) Window Center                       DS: "30"
        (0028, 1051) Window Width                        DS: "80"
        (0028, 1052) Rescale Intercept                   DS: "-1024"
        (0028, 1053) Rescale Slope                       DS: "1"
        (7fe0, 0010) Pixel Data                          OW: Array of 524288 elements
    '''

    '''As we can see, each image stores a significant amount of image such as the ID called "UI" and much more. 
    The only thing that is necessary for this investigation is the "Pixel Data" so next, well create a varible which
    contains the the image's pixel data'''

    # Defining "pixelArray" and setting it equal to the "ds" object pixel_array
    pixelArray = ds.pixel_array

    # Lets have a look at what the first 5 rows of the pixel array looks like by paining it out
    print(pixelArray[0:5]) if n == 0 else None

    '''
        Output:
        [[-2000 -2000 -2000 ... -2000 -2000 -2000]
         [-2000 -2000 -2000 ... -2000 -2000 -2000]
         [-2000 -2000 -2000 ... -2000 -2000 -2000]
         [-2000 -2000 -2000 ... -2000 -2000 -2000]
         [-2000 -2000 -2000 ... -2000 -2000 -2000]]
        
        The pixelArray is an array consisting an array for each row of the image, so if the image is 512 pixels wide, 
        there will be 512 arrays (height) consisting of 512 pixel values (width).
    '''

    # Display the shape of the array
    print(pixelArray.shape) if n == 0 else None
    ''' 
        Output:
        (512, 512)
    '''

    '''The original images that were extracted have dimensions 512x512 which is too large for this investigation. 
    Therefore, a function called resize within the sklearn (skimage) library is used to resize the pixel_array to a 100 by 100
    image.'''


    # Lets redefine the "pixelArray" varible by applying the resize function to it
    # Recall that "PIXEL_SIZE" is set to 100 so the new array should have shape (100, 100)
    pixelArray = resize(pixelArray, (PIXEL_SIZE, PIXEL_SIZE))

    '''Lets make sure that the the pixel array has been successfully converted by raising and error if it has not been'''

    # If the pixelArray shape is not equal to (100, 100) then raise an error
    if pixelArray.shape != (PIXEL_SIZE, PIXEL_SIZE):
        raise Exception(f"Pixel Array shape not equal to ({PIXEL_SIZE}, {PIXEL_SIZE}), it is equal to {pixelArray.shape}")

    '''
        Expected output in case of error:
        Pixel Array shape not equal to (100, 100}, it is equal to |Shape of pixelArray|
    

    Having a dictionary with 110000 entries requires a lot of memory, therefore, memory optimisation tasks have
    to be performed in order to minimize the strain on the computer and allow the program to operate faster. There are
    different ways this can be done, the chose that will be used for this program is creating multiple files that will
    store 5000 values at a time and save them to a .pickle file. Therefore, there will be 22 files in total each storing
    the 110000 thousand pixel arrays and labels.
    
    
    Lets start by checking if 5000 images have been analysed by using the varible "n" (the index number of the current
    image).
    '''

    # If n mod 5000 = 0 then clear the current "dataDivided" dictionary and set it equal to an empty one.
    # Also, this should not be run on the last image since this would empty the dataDivided dict before the saving file
    if n % 5000 == 0 and n != 110000:
        # Save the "dataDivided" dictionary and name the file train_data_"n/5000" where n/5000 is number of the file that are
        # to be created (22 in total)
        pickle.dump(dataDivided, open(f"GYARprocessedData/train_data_{int(n/5000)}.pickle", "wb")) if dataDivided and int(n/5000) != 22 else None

        print(f"File number {int(n/5000)} has been written...")

        # Clearing the "dataDivided" dictionary
        dataDivided = {}




    '''The next step is to insert the pixel array into the "data" dictionary created earlier. The output/defenition of
    each instance in the dictionary should be an array in itself where the first element is the pixel array of the image
    and the second element is the label (the definition that already exists)'''

    # Redefining the definition of each key (imageID) in the "data" dictionary
    # On the right hand side of the = sign, data[key] will retrieve the current values of the dictionary.
    # Create a new key: defenition in the "dataDivided" dictionary and setting it equal to the pixel array and the label
    dataDivided[image] = [pixelArray, data[image]]


    if n % 1000 == 0:
        print(f"{np.round((n/110000)*100, decimals=1)} % of images have been processed...")
        print(len(data))

    # Remove the entry from the "data" dictionary
    del data[image]

    # When the last 5000 entries in the "data" dictionary have been processed, it will instead be saved as evaluate
    # data instead of training. Also it should write the file if there is already 22 files present
    if n == 109999 and numFiles != 22:
        pickle.dump(dataDivided, open(f"GYARprocessedData/evaluate_data_{int(n/5000)+1}.pickle", "wb"))

print("Training/evaluation files successfully completed...")

'''Now that the training and evaluation data is prepared, the next step will be to build the neural network which
will be able to output a number between 0 and 1. 0 indicates no IH and 1 indicated present IH. The algorithms job is to
optimize its weights and biases in order to maximise the accuracy score and minimise the loss function by getting closer 
to 0 and 1 depending on the label'''

'''First of all, a library called tensorflow will be used for this step. It is a module made by google which uses an API
called Keras to create neural networks'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

tf.get_logger().setLevel('INFO')


physical_devices = tf.config.experimental.list_physical_devices(device_type=None)
print(tf.test.is_gpu_available())
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

'''When a model is after it has been trained, the weights need to correspond to the same size neural network as it was
trained on, otherwise and error will occur. This is why the naming of each model file will contain specific information
about the model's size, nodes, functions etc. This is so, at a later date, the model can be loaded in an evaluated'''

'''But before we create the model itself, we need to prepare the training data. As the training data has been saved to
22 separate files, they have to be loaded in again. However, some computers cannot handle 110000 files within its memory,
which is why the neural network will be trained 2 different times, one with 50000 files and the other with 55000, where
the remaining 5000 files will be used to evaluate the accuracy of the model.'''

# Creating a function which loads in a specified number of files
def loadInTrainingData(FromNumFiles, ToNumFiles):
    TotalNumberOfFiles = (ToNumFiles - FromNumFiles)*5000

    # Defining the X-data (the pixel array) and the y-data (the label either 0 or 1)
    y, X = np.zeros((TotalNumberOfFiles, 1)), np.zeros((TotalNumberOfFiles, PIXEL_SIZE, PIXEL_SIZE), dtype=np.float32)

    # Setting the index in which place the next entry will be inserted into the arrays above to 0
    index = 0

    # Iterating over all number from 1 to 21, which are the number of training data files that exist.
    for i in range(FromNumFiles, ToNumFiles):
        # Setting the filename
        file = f"GYARprocessedData/train_data_{i}.pickle" if FromNumFiles != 22 else f"GYARprocessedData/evaluate_data_{i}.pickle"
        # Loading in the file
        loadedData = pickle.load(open(file, "rb"))
        # Iterating over each entry in the dictionary
        for key in loadedData.keys():
            # Adding the first element of the key defenition (pixel array) to the X array (input)
            X[index] = loadedData[key][0]
            # Adding the second element of the key defenition (label) to the y array (expected output)
            y[index] = loadedData[key][1]
            # Incramenting index by 1
            index+=1

            # Exit function and return X and y if the index == to the number of files
            if index == TotalNumberOfFiles:
                return X, y



'''No we have to ask weather the program should train the neural network or evaluate it we do this by using the python
function "input" which will require the user to write something when the program reaches this line of code'''
run = None

while True:
    run = input("Train a new neural Network? (y/n): ")
    if run == 'y' or run == 'n':
        break


if run == 'y':
    Train = True
    Evaluate = False
else:
    Train = False
    # If the user does not want to train a new model, ask if the user wants to evaluate already existing models
    while True:
        run = input("Evaluate Models? (y/n): ")
        if run == 'y' or run == 'n':
            break
    if run == 'y':
        Evaluate = True
    else:
        Evaluate = False
        print("Nothing has been selected, exiting program")
        #TODO put back exit
        #exit()


'''Now, the create model function can be created. It will take in several argumnets such as the amount of convolutional
layers, the size of the convolutional layer, the amount of fully connected layers, the size of the fully connected layers,
the batch size, the optimizer function, the loss function that should be used and the number of epochs to train the 
network for. The output will be the created model'''

def createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER="adam", LOSS_FUNCTION="binary_crossentropy", EPOCHS=2):
    # The first step will be creating the file name. The file name will contain all the relevant information on the
    # neural network modal so it can be replicated at a later date for evaluation.
    lf = LOSS_FUNCTION.replace("_", "-")
    global MODEL_NAME
    MODEL_NAME = f"R-gyar-model_convs-{CONV_LAYERS}_convnodes-{CONV_LAYER_SIZE}_denses-{DENSE_LAYERS}_densenodes-{DENSE_LAYER_SIZE}_batch-{BATCH_SIZE}_opt-{OPTIMIZER}_loss-{lf}_epochs-{EPOCHS}_time-{round(time.time())}"

    global tensorboard
    # Creating tensorboard object, this can be used to view metrics and graphs of how the well the neural network is training
    #tensorboard = TensorBoard(log_dir='GYARlogs\\{}'.format(MODEL_NAME))
    tensorboard = TensorBoard(log_dir='GYARlogs\\{}'.format(MODEL_NAME), histogram_freq=1, profile_batch=3)

    # If the model already exist raise a file exists error else, create the model.
    if MODEL_NAME in os.listdir('GYARlogs'):
        raise FileExistsError
    else:
        if Train:
            print("Model Created: ", MODEL_NAME)

        # Initializing a Sequential model, this is the basis for the model on which further layers can be added.
        model = Sequential()

        # Adding a convolutional2D layer with input shape of an image (100 by 100)
        model.add(Conv2D(CONV_LAYER_SIZE, (3, 3), input_shape=(100, 100, 1)))

        # Activation function will return 1 for all x > 0 and return 0 for x <= 0
        model.add(Activation('relu'))

        # Maxpooling2D is required for convolutional neural network since it takes the maximum value of a 2 by 2 area
        # and created a new image of it.
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Repeat this step for how ever many convlolutional layers where specified in the input
        for l in range(CONV_LAYERS-1):
            model.add(Conv2D(CONV_LAYER_SIZE, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        # this converts our 3D feature maps to 1D feature vectors, basically takes the (100, 100) pixel array and makes
        # a 1D array with size (10000) instead (100 times 100)
        model.add(Flatten())

        # Create how ever many fully connected (dense) layers as specified in the input
        for l in range(DENSE_LAYERS):
            model.add(Dense(DENSE_LAYER_SIZE))

            # Add the relu activation function
            model.add(Activation('relu'))

            # Adding dropout which will cause some perceptrons in the neural net to deactivate
            model.add(Dropout(0.5))


        # Add a final fully connected layer with size one, this is our output
        model.add(Dense(1))

        # Adding the sigmoid activation function which will return a value between 0 and 1, corresponding to prediction
        # for the image, 0 for negative IH and 1 for positive IH
        model.add(Activation('sigmoid'))

        # Returning the model (output)
        return model




'''Now that the createmodel function has been created, a trainmodel function needs to be set up. This is where training
actually takes place and where the images are inserted into the neural network and the neural network is able to change
its weights and biases to maximize accuracy and minimize loss. '''

# "train_model" function takes in two inputs, which is our X and y that was defined earlier, input_data is the X and
# the output_data (labels) is the y
def train_model(input_data, output_data):

    # NETWORKVARIABLES
    # Here, arrays of each varible is made in order to create different types of neural nets. Example, if the array EPOCHS
    # Had values 5 and 10, a new model would be created for each entry in the array, meaning that two models will be created,
    # One that has been trained for 5 epochs and the other which has been trained for 10. In this way, multiple variants
    # of the neural network can be trained. The amount of variants (permutations) of neural nets can be calculated by
    # multiplying the length of each array which one another. If all the arrays have length one, only on model will be
    # created.
    CONV_LAYERS = [1,2,3]
    CONV_LAYER_SIZE = [10, 20]
    DENSE_LAYERS = [1,2,3]
    DENSE_LAYER_SIZE = [32, 64, 128]
    BATCH_SIZE = [64, 128]
    EPOCHS = [10]
    OPTIMIZERS = ["Nadam", "adam"]
    LOSS_FUNCTIONS = ["huber_loss", "binary_crossentropy"]

    # The number of models is number of combinations that exist
    number_of_models = len(CONV_LAYERS)*len(CONV_LAYER_SIZE)*len(DENSE_LAYERS)*len(DENSE_LAYER_SIZE)*len(BATCH_SIZE)*len(EPOCHS)*len(OPTIMIZERS)*len(LOSS_FUNCTIONS)
    print(f"Number of models that will be created is: {number_of_models}")

    # Iterating over each entry in the different arrays.
    for conv_layers in CONV_LAYERS:
        for conv_layer_size in CONV_LAYER_SIZE:
            for dense_layers in DENSE_LAYERS:
                for dense_layer_size in DENSE_LAYER_SIZE:
                    for batch in BATCH_SIZE:
                        for optimizer in OPTIMIZERS:
                            for loss_function in LOSS_FUNCTIONS:
                                for epochs in EPOCHS:
                                    # Creating the model, based on the current set of values (lower case varibles)
                                    model = createModel(CONV_LAYERS=conv_layers, CONV_LAYER_SIZE=conv_layer_size, DENSE_LAYERS=dense_layers,
                                                        DENSE_LAYER_SIZE=dense_layer_size, BATCH_SIZE=batch, OPTIMIZER=optimizer, LOSS_FUNCTION=loss_function, EPOCHS=epochs)

                                    # Compile the model. Sets it up for being trained.
                                    model.compile(loss=loss_function,
                                                  optimizer=optimizer,
                                                  metrics=['accuracy'])
                                    # Lets have a look at the model summary. Keep in mind that this might change depending on
                                    # The network variables defined before
                                    print(model.summary())
                                    '''
                                        Output:
                                        _________________________________________________________________
                                        Layer (type)                 Output Shape              Param #   
                                        =================================================================
                                        conv2d (Conv2D)              (None, 98, 98, 10)        100       
                                        _________________________________________________________________
                                        activation (Activation)      (None, 98, 98, 10)        0         
                                        _________________________________________________________________
                                        max_pooling2d (MaxPooling2D) (None, 49, 49, 10)        0         
                                        _________________________________________________________________
                                        flatten (Flatten)            (None, 24010)             0         
                                        _________________________________________________________________
                                        dense (Dense)                (None, 75)                1800825   
                                        _________________________________________________________________
                                        activation_1 (Activation)    (None, 75)                0         
                                        _________________________________________________________________
                                        dropout (Dropout)            (None, 75)                0         
                                        _________________________________________________________________
                                        dense_1 (Dense)              (None, 75)                5700      
                                        _________________________________________________________________
                                        activation_2 (Activation)    (None, 75)                0         
                                        _________________________________________________________________
                                        dropout_1 (Dropout)          (None, 75)                0         
                                        _________________________________________________________________
                                        dense_2 (Dense)              (None, 1)                 76        
                                        _________________________________________________________________
                                        activation_3 (Activation)    (None, 1)                 0         
                                        =================================================================
                                        Total params: 1,806,701
                                        Trainable params: 1,806,701
                                        Non-trainable params: 0
                                        _________________________________________________________________

                                    '''

                                    print("Commencing Training...")
                                    # The fit function inserts the training data in batches, specified by the batch_size varible,
                                    # It trains for a number of epochs specified by the "epochs" varible and it shuffles the
                                    # training data
                                    model.fit(input_data, output_data, batch_size=batch, epochs=epochs,
                                              callbacks=[tensorboard], shuffle=True)

                                    del input_data
                                    del output_data
                                    # Now that the model has been trained on the first portion of the training data (50000 images),
                                    # it will be trained on the remaining 55000 images, which is file 11 to 21, 11 files in total,
                                    # 22 not included
                                    input_data, output_data = loadInTrainingData(11, 22)

                                    # Reshaping the array just a before
                                    input_data = np.reshape(input_data, (len(input_data), PIXEL_SIZE, PIXEL_SIZE, 1))

                                    # Fit the neural network with the new training data, the remaining 55000 images
                                    model.fit(input_data, output_data, batch_size=batch, epochs=epochs,
                                              callbacks=[tensorboard], shuffle=True)

                                    # Save the model's weights and biases so it can be loaded and evaluated later
                                    model.save_weights('GYARmodels/{}.h5'.format(MODEL_NAME))


'''Now that the trainmodel function has been created, a evaluate function needs to be created. This function will evaluate 
a chosen neural network or all existing neural networks using the 22nd file which is named evaluate_data. 

For this step, a class will be used instead of a function0

'''

import json
# This function will be used to train the 20 best models from all the models that will be trained
def train_model_from_input(filepath):
    d = json.load(open(filepath, "r"))
    epochs = 20
    for model_params in d.values():
        input_data, output_data = loadInTrainingData(1, 11)
        input_data = np.reshape(input_data, (len(input_data), PIXEL_SIZE, PIXEL_SIZE, 1))

        conv_layers = model_params["conv_layers"]
        conv_layer_size = model_params["conv_nodes"]
        dense_layers = model_params["dense_layers"]
        dense_layer_size = model_params["dense_nodes"]
        batch = model_params["batch_size"]
        optimizer = model_params["optimizer"]
        loss_function = model_params["loss_function"]


        model = createModel(CONV_LAYERS=conv_layers, CONV_LAYER_SIZE=conv_layer_size, DENSE_LAYERS=dense_layers,
                            DENSE_LAYER_SIZE=dense_layer_size, BATCH_SIZE=batch, OPTIMIZER=optimizer,
                            LOSS_FUNCTION=loss_function, EPOCHS=epochs)

        model.compile(loss=loss_function,
                      optimizer=optimizer,
                      metrics=['accuracy'])


        print("Commencing Training...")
        model.fit(input_data, output_data, batch_size=batch, epochs=epochs,
                  callbacks=[tensorboard], shuffle=True)

        del input_data
        del output_data
        input_data, output_data = loadInTrainingData(11, 22)

        input_data = np.reshape(input_data, (len(input_data), PIXEL_SIZE, PIXEL_SIZE, 1))

        model.fit(input_data, output_data, batch_size=batch, epochs=epochs,
                  callbacks=[tensorboard], shuffle=True)

        model.save_weights('GYARmodels/{}.h5'.format(MODEL_NAME))



import shutil
class EvaluateHandler:
    fileLocation = "GYARmodels"

    def __init__(self, minago=None):
        
        self.evaluate_X_data, self.evaluate_y_data = loadInTrainingData(22, 23)
        self.evaluate_X_data = np.reshape(self.evaluate_X_data, (len(self.evaluate_X_data), PIXEL_SIZE, PIXEL_SIZE, 1))

        self.filedata = {}
        self.models = {}
        n = 0
        if minago != None:
            for model in os.listdir(self.fileLocation):
                if int(model.split("_time-")[1][:-3]) > time.time()-minago*60:
                    self.models[n] = model
                    n+=1
        else:
            self.models = {n: model for n, model in enumerate(os.listdir(self.fileLocation))}


    def choseModel(self):
        print("\n|------------------------------------------------------|\n")
        for key in self.models:
            print(f"[{key}] {self.models[key]}")
        print("[-1] All Models")
        try:
            modelNumber = int(input("\nEnter the model number you would like to evaluate (-1 for all models): "))
        except:
            print("\nInvalid Number, Try again...\n")
            self.choseModel()
        print(f"You selected model number {modelNumber}")
        print("\n|------------------------------------------------------|\n")

        if modelNumber == -1:
            return -1
        else:
            return self.models[modelNumber]

    def initCSV(self, file_name):
        with open(os.path.join("GYAREvaluations", f"{file_name}.csv"), "w") as f:
            f.write("loss,accuracy,0 true,1 true,conv_layers,conv_nodes,dense_layers,dense_nodes,optimizer,loss_function,epoch,batch_size\n")
            f.close()


    def generateDataFile(self):
        os.mkdir("GYAREvaluations") if not os.path.exists("GYAREvaluations") else None

        print(self.filedata)
        file_name = f"entries-{len(self.filedata.keys())}-{int(time.time())}"
        self.initCSV(file_name)

        with open(os.path.join("GYAREvaluations", f"entries-{len(self.filedata.keys())}-{int(time.time())}.csv"), "a") as f:
            for model in self.filedata.values():
                f.write(f"{model['Loss']},{model['Accuracy']},{model['Distribution']['0 true']},{model['Distribution']['1 true']},"
                        f"{model['Model Info']['Conv. Layers']},"f"{model['Model Info']['Conv. Nodes']},"
                        f"{model['Model Info']['Dense Layers']},{model['Model Info']['Dense Nodes']},{model['Model Info']['Optimizer']},"
                        f"{model['Model Info']['Loss Function']},{model['Model Info']['Epochs Trained']},{model['Model Info']['Batch Size']}\n")
            f.close()



    def Evaluate(self):
        modelname = self.choseModel()

        if modelname == -1:
            print("\n|*** Evaluating selected models! ***|\n")
            for n, modelname in enumerate(self.models.values()):
                self.filedata[n] = self._eval(modelname)

            self.generateDataFile()
        else:
            self._eval(modelname)

        print("\n|*** Evaluation Complete ***|\n")

    def _strip_modelname(self, modelname):
        CONV_LAYERS = int(modelname.split("convs")[1][1:].split("_")[0])
        CONV_LAYER_SIZE = int(modelname.split("convnodes")[1][1:].split("_")[0])
        DENSE_LAYERS = int(modelname.split("denses")[1][1:].split("_")[0])
        DENSE_LAYER_SIZE = int(modelname.split("densenodes")[1][1:].split("_")[0])
        BATCH_SIZE = int(modelname.split("batch")[1][1:].split("_")[0])
        EPOCHS = int(modelname.split("epochs")[1][1:].split("_")[0])

        try:
            OPTIMIZER = str(modelname.split("opt")[1][1:].split("_")[0])
            LOSS_FUNCTION = str(modelname.split("loss", 1)[1][1:].split("_")[0]).replace("-", "_")
        except:
            OPTIMIZER = "adam"
            LOSS_FUNCTION = "binary_crossentropy"

        return CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS_FUNCTION


    def _eval(self, modelname):
        CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS_FUNCTION = self._strip_modelname(modelname)

        networkDetails = f"CONVS {CONV_LAYERS}, CONVNODES {CONV_LAYER_SIZE}, DENSE LAYERS {DENSE_LAYERS}, DENSENODES {DENSE_LAYER_SIZE}, BATCH {BATCH_SIZE}, OPTIMIZER {OPTIMIZER}, LOSS {LOSS_FUNCTION}, EPOCHS {EPOCHS}"
        print(networkDetails)

        model = createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER,
                            LOSS_FUNCTION, EPOCHS)
        model.load_weights('GYARmodels/{}'.format(modelname))
        model.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])

        loss, acc = model.evaluate(self.evaluate_X_data, self.evaluate_y_data)

        predictions = model.predict(self.evaluate_X_data)

        predictions = np.array(predictions)
        save_predictions = np.array(list(zip(predictions, self.evaluate_y_data)))
        np.save(f"GYARPredictions/{acc}_{loss}_{int(time.time())}.npy", save_predictions)


        correct = [(self.evaluate_y_data[x], predictions[x], np.round(predictions[x], decimals=0) == self.evaluate_y_data[x]) for x in range(len(self.evaluate_y_data))]
        displayValues = False
        if displayValues:
            for x in correct:
                print(x[0], x[1], x[2])


        a = list(map(lambda x: x[0], correct))

        unique, counts = np.unique(a, return_counts=True)
        print(dict(zip(unique, counts)))

        dis = {"0 true": 0, "1 true": 0}

        for x in correct:
            if x[0] == 0 and x[2] == True:
                dis["0 true"] += 1
            elif x[0] == 1 and x[2] == True:
                dis["1 true"] += 1



        return {"Loss": round(loss, 6), "Accuracy": acc, "Distribution": dis, "Model Info": {"Conv. Layers": CONV_LAYERS,
                                                                                   "Conv. Nodes": CONV_LAYER_SIZE,
                                                                                   "Dense Layers": DENSE_LAYERS,
                                                                                   "Dense Nodes": DENSE_LAYER_SIZE,
                                                                                   "Optimizer": OPTIMIZER,
                                                                                   "Loss Function": LOSS_FUNCTION,
                                                                                   "Epochs Trained": EPOCHS,
                                                                                   "Batch Size": BATCH_SIZE}}


    def _ids_and_labels(self):
        il = pickle.load(open("IDandLabels.pickle", "rb")) if os.path.isfile("IDandLabels.pickle") else None
        return il if il is not None else False

    def showim(self, ds, ID=None):
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.show()

    def moveDCM(self, from_path, to_path):
        shutil.copyfile(from_path, to_path)

    def evaluateDicomImage(self, ID, modelname):

        if ID not in list(evaluate_images.keys()):
            raise FileExistsError("File not found in evaluate dataset")

        place_image_dir = "GYAR Specific Image Testing"

        os.mkdir(place_image_dir) if not os.path.exists(place_image_dir) else None
        if not os.path.isfile(place_image_dir+"/keep.csv"):
            with open(place_image_dir+"/results.csv", "w") as f:
                f.write("ID,pred,actual\n")
                f.close()

        imagePath = extractImagePath + "/stage_1_train_images/" + ID + ".dcm"
        ds = pydicom.dcmread(imagePath)
        pixelArray = ds.pixel_array

        X = resize(pixelArray, (PIXEL_SIZE, PIXEL_SIZE))
        X = np.reshape(X, (1, PIXEL_SIZE, PIXEL_SIZE, 1))


        CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS_FUNCTION = self._strip_modelname(modelname)
        model = createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER,
                            LOSS_FUNCTION, EPOCHS)
        model.load_weights('GYARmodels/{}'.format(modelname))
        model.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])

        pred = model.predict(X)


        print(ID, pred[0][0], evaluate_images[ID])

        from_path = "GYARImageData/stage_1_train_images/" + ID + ".dcm"
        to_path = place_image_dir + "/" + ID + ".dcm"
        self.moveDCM(from_path, to_path)

        shutil.copyfile(to_path, place_image_dir+"/result/"+ID+".dcm")
        os.remove(to_path)
        print(f"Keeping {ID}...")
        with open(place_image_dir + "/results.csv", "a") as f:
            f.write(f"{ID},{np.round(pred[0][0], 5)},{evaluate_images[ID]}\n")
            f.close()


def askToDo(text, confirm_str, no_str):
    ans = ""
    tries = 0
    while ans != confirm_str:
        ans = input(text)
        tries += 1
        if tries == 3 or ans == no_str:
            return False
    return True



#
# # If Train is set to True, commence training
if Train:
    # Creating X, y training data consisting of 50000 which is 10 files, hence file 1 to 10, 11 is not included
    X, y = loadInTrainingData(1, 11)

    # Reshaping the X array to fit the neural network later
    X = np.reshape(X, (len(X), PIXEL_SIZE, PIXEL_SIZE, 1))

    # Lets have a look at the shape of the both arrays
    print(X.shape, y.shape)
    '''
        Output:
        X -> (50000, 100, 100, 1) y -> (50000, 1)

    '''
    # Now, lets call the train_model function and insert the X and y data previously created
    train_model(X, y)
if Evaluate:
    minago = input("How many minutes ago should the models be picked from? (enter int, -1 for all models): ")

    if minago == "-1":
        minago = None
    else:
        try:
            minago = int(minago)
        except Exception as e:
            print("Error: ", e)
            minago=None
    EHandler = EvaluateHandler(minago)
    ask_to_evaluate_images = askToDo("Do you want to evaluate each image individually? (y/n):", "y", "n")
    if ask_to_evaluate_images:
        for ID in list(evaluate_images.keys()):
            EHandler.evaluateDicomImage(ID, "R-gyar-model_convs-2_convnodes-20_denses-2_densenodes-128_batch-128_opt-Nadam_loss-huber-loss_epochs-15_time-1579187453.h5")
    EHandler.Evaluate()


