import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from matplotlib import style
import json

style.use("seaborn-bright")


def loadIDLabelData():
    font = {'family': 'arial',
            'size': 15}

    plt.rc('font', **font)

    data = pickle.load(open("IDandLabels.pickle", "rb"))
    data = list(map(lambda x: x[1], list(data.items())))
    data = list(filter(lambda x: x == 1 or x == 0, data))

    print(data)

    plt.hist(data, bins=3, align="mid", rwidth=1)
    plt.xticks(np.arange(-1, 3, 1), ["", "Positive", "Negative", ""])

    plt.gca()
    plt.grid(axis="y")
    plt.ylabel("Number of images", fontsize=20)
    plt.xlabel("Label", fontsize=20)
    plt.title("Positively & negatively labelled data", fontsize=22)

    plt.show()


def lossVsAccuracy(df):
    for k, row in df.iterrows():
        color = "r" if row["accuracy"] > 0.8 else "b"
        plt.scatter(row["loss"], row["accuracy"], c=color)
    plt.show()


def getBestScore(df, sortby, num_top_values):

    assert sortby in df.columns

    df = df.sort_values(sortby, ascending=False)

    return df[:num_top_values]

def compareVariblesToAccuracy(df, column):
    for denses in list(set(df[column])):
        rows = df.loc[df[column] == denses]

        mean = rows["accuracy"].mean()
        error = (max(rows["accuracy"]) - min(rows["accuracy"]))/2

        plt.errorbar(denses, mean, error, ecolor="k", elinewidth=2, fmt="o")

    plt.title(f"{column} V accuracy")
    plt.show()

def compareVariblesToVarible(df, column1, column2):
    for x in list(set(df[column1])):
        rows = df.loc[df[column1] == x]

        mean = rows[column2].mean()
        error = (max(rows[column2]) - min(rows[column2]))/2
        plt.errorbar(x, mean, yerr=error, ecolor="k", elinewidth=2, fmt="o", dash_joinstyle="round")

    plt.title(f"{column1} V {column2}")
    plt.show()

def saveBestModels(df, num):
    df = getBestScore(df, "accuracy", num)

    d = {}

    for k, row in df.iterrows():
        row = row.drop(["loss", "accuracy", "0 true", "1 true", "epoch"])
        d[k] = row.to_dict()

    json.dump(d, open("bestModels.json", "w"), indent=2)





def loadEvaluation(filepath):
    df = pd.read_csv(filepath)

    # Getting rid of training with 15 epochs
    #df = df.drop(302)

    df = df.sort_values("accuracy")
    bestAccuracies = getBestScore(df, "accuracy", 20)
    print(bestAccuracies)

    saveBestModels(df, 20)



    for column in df.columns:
        try:
            pass
            #compareVariblesToVarible(df, column, "accuracy")
        except:
            pass










loadEvaluation("GYAREvaluations/entries-20-1580330613.csv")
