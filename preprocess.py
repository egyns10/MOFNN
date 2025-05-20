# preprocess.py
import pandas as pd
import numpy as np

def readCSV(filepath):
    readData = pd.read_csv(filepath)
    return readData

def removeDup(data):
    noDup = data.drop_duplicates(keep='first').reset_index(drop=True)
    return noDup

def cleanData(data):
    noString = data.map(lambda x: x if (isinstance(x, (int, float, np.number)) and not pd.isnull(x)) else np.nan)
    # noNull = noString.dropna().reset_index(drop=True)
    noNull = noString.dropna(axis=1, how='all').reset_index(drop=True)
    return noNull

def saveAsCSV(data, filePath):
    data.to_csv(filePath, index=False)

def take2Col(data, colX, colY):
    colX = int(colX)
    colY = int(colY)
    isolatedCol = data.iloc[:, [colX, colY]]
    return isolatedCol