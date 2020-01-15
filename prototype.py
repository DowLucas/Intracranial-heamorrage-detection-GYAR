import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time
import random
import os
import argparse
import csv

# Import tensorflow.keras modules
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

SUB_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

parser = argparse.ArgumentParser(description='Train or not?')

parser.add_argument('-train', metavar='T', type=bool, help='do you want to train a new model?')
parser.add_argument("-model", metavar='M', type=str, help='Select a model to evaluate on')
parser.add_argument("-repeat", metavar='R', type=bool, help='If set true, will evaluate over every model in folder.')

args = parser.parse_args()

Repeat = True if args.repeat else False
Train = True if args.train else False

PIX_SIZE = 100

def loadTrainImages():

    # DATA
    data_file = "all_data.pickle"
    time1 = time.time()
    print("Loading Train Images. Please wait...")
    data = pickle.load(open("train/"+data_file, "rb"))
    #y = pickle.load(open("train/labels.pickle", "rb"))
    print("{} Entries Loaded. Time taken: {} s\n".format(len(data), round(time.time()-time1)))

    #print(data)

    assert len(data) != 0, "Data not imported, length = 0"

    X = np.zeros((len(data), PIX_SIZE, PIX_SIZE))
    y = np.zeros((len(data), 1))

    for n,key in enumerate(data.keys()):
        X[n] = data[key][0]
        y[n] = data[key][1]

    X = X.reshape((len(data), PIX_SIZE, PIX_SIZE, 1))

    return X, y

X, y = loadTrainImages()

def loadEvalImages():

    # Load evalute images
    print("Loading evalute images...")
    evaluate_images = pickle.load(open("train/evaluate_data.pickle", "rb"))

    X_test = np.zeros((len(evaluate_images), PIX_SIZE, PIX_SIZE))
    y_test = np.zeros((len(evaluate_images), 1))

    for n, key in enumerate(evaluate_images.keys()):
        X_test[n] = evaluate_images[key][0]
        y_test[n] = evaluate_images[key][1]

    X_test = X_test.reshape((len(evaluate_images), PIX_SIZE, PIX_SIZE, 1))
    return X_test, y_test

X_test, y_test = loadEvalImages()

assert X.shape == (len(X), PIX_SIZE, PIX_SIZE, 1), "X.shape not correct"
assert y.shape == (len(X), 1), "y.shape not correct"

distrubution = {"pos": 0, "neg": 0}

for en in y:
    if en[0] == 1:
        distrubution["pos"] += 1
    elif en[0] == 0:
        distrubution["neg"] += 1

print(distrubution)

# Tensorboard names

def createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER="adam", LOSS_FUNCTION="binary_crossentropy", EPOCHS=2):
    lf = LOSS_FUNCTION.replace("_", "-")
    global MODEL_NAME
    MODEL_NAME = f"R-gyar-prototype-model_convs-{CONV_LAYERS}_convnodes-{CONV_LAYER_SIZE}_denses-{DENSE_LAYERS}_densenodes-{DENSE_LAYER_SIZE}_batch-{BATCH_SIZE}_opt-{OPTIMIZER}_loss-{lf}_epochs-{EPOCHS}_time-{round(time.time())}"
    global tensorboard
    tensorboard = TensorBoard(log_dir='prototypelogs/{}'.format(MODEL_NAME))
    if MODEL_NAME in os.listdir('prototypelogs'):
        raise FileExistsError
    else:
        if Train:
            print("Model Created: ", MODEL_NAME)

        model = Sequential()

        model.add(Conv2D(CONV_LAYER_SIZE, (3, 3), input_shape=(100, 100, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for l in range(CONV_LAYERS-1):
            model.add(Conv2D(CONV_LAYER_SIZE, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        for l in range(DENSE_LAYERS):
            model.add(Dense(DENSE_LAYER_SIZE))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))


        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
def train_model():

    # NETWORKVARIABLES

    CONV_LAYERS = [1]
    CONV_LAYER_SIZE = [10]
    DENSE_LAYERS = [2]
    DENSE_LAYER_SIZE = [75]
    BATCH_SIZE = [64]
    EPOCHS = [10]
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "binary_crossentropy"


    for conv_layers in CONV_LAYERS:
        for conv_layer_size in CONV_LAYER_SIZE:
            for dense_layers in DENSE_LAYERS:
                for dense_layer_size in DENSE_LAYER_SIZE:
                    for batch in BATCH_SIZE:
                        for epochs in EPOCHS:
                            model = createModel(conv_layers, conv_layer_size, dense_layers, dense_layer_size, batch, OPTIMIZER, LOSS_FUNCTION, epochs)
                            model.compile(loss=LOSS_FUNCTION,
                                          optimizer=OPTIMIZER,
                                          metrics=['accuracy'])

                            print("Commencing Training...")
                            model.fit(X, y, batch_size=batch, epochs=epochs, callbacks=[tensorboard], shuffle=True)
                            model.evaluate(X_test, y_test)
                            model.save_weights('prototypemodels/{}.h5'.format(MODEL_NAME))


def correct(pred, actual):
    if round(pred[0]) == actual[0]:
        return 1
    else:
        return 0

def simpleEvalute(modelname):
    print(modelname)
    CONV_LAYERS = int(modelname.split("convs")[1][1:].split("_")[0])
    CONV_LAYER_SIZE = int(modelname.split("convnodes")[1][1:].split("_")[0])
    LAYER_SIZE = int(modelname.split("nodes")[1][1:].split("_")[0])
    DENSE_LAYERS = int(modelname.split("denses")[1][1:].split("_")[0])
    DENSE_LAYER_SIZE = int(modelname.split("densenodes")[1][1:].split("_")[0])
    BATCH_SIZE = int(modelname.split("batch")[1][1:].split("_")[0])
    EPOCHS = int(modelname.split("epochs")[1][1:].split("_")[0])

    if not CONV_LAYER_SIZE or not DENSE_LAYER_SIZE:
        CONV_LAYER_SIZE = LAYER_SIZE
        DENSE_LAYER_SIZE = LAYER_SIZE
    try:
        OPTIMIZER = str(modelname.split("opt")[1][1:].split("_")[0])
        LOSS_FUNCTION = str(modelname.split("loss")[1][1:].split("_")[0]).replace("-", "_")
    except:
        print("No Optimizer or loss function found!")
        OPTIMIZER = "adam"
        LOSS_FUNCTION = "binary_crossentropy"

    model = createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, EPOCHS)
    model.load_weights('prototypemodels/{}'.format(modelname))

    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])
    loss, acc = model.evaluate(X_test, y_test)

    with open('modelsaccuracy.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(CONV_LAYERS), str(CONV_LAYER_SIZE), str(DENSE_LAYERS), str(DENSE_LAYER_SIZE), str(BATCH_SIZE), str(EPOCHS), OPTIMIZER, LOSS_FUNCTION, acc, loss])





def Evaluate():
    if Repeat:
        for model in os.listdir("prototypemodels"):
            if ".h5" in model:
                simpleEvalute(model)
                print("---------------------------------------------------------")


    modelname = input("Insert Model name, excluding extension (type 'q' to quit): ")

    if modelname == 'q':
        quit()

    CONV_LAYERS = int(modelname.split("convs")[1][1:].split("_")[0])
    CONV_LAYER_SIZE = int(modelname.split("convnodes")[1][1:].split("_")[0])
    LAYER_SIZE = int(modelname.split("nodes")[1][1:].split("_")[0])
    DENSE_LAYERS = int(modelname.split("denses")[1][1:].split("_")[0])
    DENSE_LAYER_SIZE = int(modelname.split("densenodes")[1][1:].split("_")[0])
    BATCH_SIZE = int(modelname.split("batch")[1][1:].split("_")[0])
    EPOCHS = int(modelname.split("epochs")[1][1:].split("_")[0])

    if not CONV_LAYER_SIZE or not DENSE_LAYER_SIZE:
        CONV_LAYER_SIZE = LAYER_SIZE
        DENSE_LAYER_SIZE = LAYER_SIZE
    try:
        OPTIMIZER = str(modelname.split("opt")[1][1:].split("_")[0])
        LOSS_FUNCTION = str(modelname.split("loss")[1][1:].split("_")[0]).replace("-", "_")
    except:
        print("No Optimizer or loss function found!")
        OPTIMIZER = "adam"
        LOSS_FUNCTION = "binary_crossentropy"

    print(f"CONVS {CONV_LAYERS}, LAYERSIZE {LAYER_SIZE}, DENSES {DENSE_LAYERS}, BATCH {BATCH_SIZE}, OPTIMIZER {OPTIMIZER}, LOSS {LOSS_FUNCTION}")

    model = createModel(CONV_LAYERS, CONV_LAYER_SIZE, DENSE_LAYERS, DENSE_LAYER_SIZE, BATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, EPOCHS)
    model.load_weights('prototypemodels/{}.h5'.format(modelname))
    #index = random.randint(0, len(X)-100)
    #predictions = X[index:index+100]
    #pred = model.predict(predictions)
    #actual = y[index:index+100]
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])
    #for _ in range(len(pred)):
    #    packet = list(zip(pred[_], actual[_]))
        #print(packet, round(packet[0][0]) == packet[0][1])
    model.evaluate(X_test, y_test)


def runEvaluate():
    while True:
        Evaluate()
        print("---------------------------------------------------------------------")


if Train:
    train_model()
    runEvaluate()
    Repeat = False
else:
    runEvaluate()