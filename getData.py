#getData.py
import pandas as pd
import json

def createGrid(features,names,results):
    #features: user chosen features e.g. PV, VF
    #names: types of algorithms e.g. XG_RF
    #results: R2, MSE etc.
    #^ these are all pd.df

    #flatten features into one single string
    title = ', '.join(features.values.flatten().astype(str))
    
    resultName = results.iloc[:, 0].tolist()
    namesML = names.iloc[:, 0].tolist()
    
    #empty grid
    grid = [[None for _ in range(len(resultName) + 1)] for _ in range(len(namesML) + 1)]
    
    #[0,0]
    grid[0][0] = title

    #first row
    for j, rows in enumerate(resultName):
        grid[0][j + 1] = rows

    #first column
    for i, cols in enumerate(namesML):
        grid[i + 1][0] = cols

    df = pd.DataFrame(grid)
    return df,title

def filterCol(reqData, data):
    #neededCol = reqData.iloc[0].tolist()
    #gotCol = data.loc[:, data.columns.isin(neededCol)]

    #neededCol = reqData.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    #return neededCol

    return data.loc[:, data.columns.isin(reqData)]

def getParas(modelName):
    with open(f'best_{modelName}_params.json', 'r') as f:
        return json.load(f)

def saveParas(modelName, bestPara):
    with open(f'best_{modelName}_params.json', 'w') as f:
        json.dump(bestPara, f) 