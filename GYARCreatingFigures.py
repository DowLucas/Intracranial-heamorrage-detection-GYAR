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
    plt.xticks(np.arange(-1, 3, 1), ["", "Negative", "Postive", ""], fontsize=25)

    plt.gca()
    plt.grid(axis="y")
    plt.ylabel("Number of images", fontsize=30)
    plt.xlabel("Label", fontsize=30)
    plt.title("Positively & negatively labelled data", fontsize=32)

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
        error = (max(rows["accuracy"]) - min(rows["accuracy"])) / 2

        plt.errorbar(denses, mean, error, ecolor="k", elinewidth=2, fmt="o")

    plt.title(f"{column} V accuracy")
    plt.show()


def compareVariblesToVarible(df, column1, column2):
    for x in list(set(df[column1])):
        rows = df.loc[df[column1] == x]

        mean = rows[column2].mean()
        error = (max(rows[column2]) - min(rows[column2])) / 2
        plt.errorbar(x, mean, yerr=error, ecolor="gray", elinewidth=1, fmt="o", dash_joinstyle="round", c="black")

    plt.xlabel(column1, fontsize=20)
    plt.ylabel(column2, fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(f"{column1} V {column2}", fontsize=23)
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
    # df = df.drop(302)

    df = df.sort_values("accuracy")
    bestAccuracies = getBestScore(df, "accuracy", 20)
    print(bestAccuracies)

    # saveBestModels(df, 20)

    compareVariblesToVarible(df, "epoch", "accuracy")

    for column in df.columns:
        try:
            compareVariblesToVarible(df, column, "accuracy")
        except:
            pass


import pandas as pd


def loadCSV():
    return pd.read_csv("stage_1_train.csv")


def subtypes_piechart(have=False):
    df = loadCSV()
    subTypes = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    df = df.loc[df["Label"] != 0]

    print(df.head())

    subTypesCount = {type: 0 for type in subTypes} if not have else {'epidural': 2761, 'intraparenchymal': 32564,
                                                                     'intraventricular': 23766, 'subarachnoid': 32122,
                                                                     'subdural': 42496, 'any': 97103}

    if not have:
        for n, row in df.iterrows():
            value = int(row[1])
            if value == 0:
                continue
            sType = row[0].split("_")[-1]

            subTypesCount[sType] += 1

    print(subTypesCount)

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', "#95ADB6", "#E5D0CC"]

    labels = [f"{stype}\n(n={subTypesCount[stype]})" for stype in subTypesCount.keys()]
    patches, texts, autotexts = plt.pie(subTypesCount.values(), labels=labels, textprops=dict(fontsize=20),
                                        autopct='%1.1f%%', colors=colors)
    for text in autotexts:
        text.set_color('black')
        text.set_fontsize(15)
    plt.title("Subtypes of Haemorrhage found within the Dataset", fontsize=25)
    plt.show()


# subtypes_piechart(True)


# loadIDLabelData()
df = loadEvaluation("GYAREvaluations/entries-496-1580315077.csv")
print(df)


