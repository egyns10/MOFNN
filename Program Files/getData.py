#getData.py
import os
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

"""
def getParas(modelName, targetProperty, combo):
    base_dir = os.path.dirname(__file__)
    jsonPath = os.path.join(base_dir, f'best_{modelName}_for_{targetProperty}_params.json')
    if not os.path.exists(jsonPath):
            return {}                   #return empty
    with open(jsonPath, 'r') as f:
            data = json.load(f)
    #combo into string key
    if isinstance(combo, tuple):
        feature_key = ','.join(combo)
    else:
        feature_key = ','.join(combo.columns)
    return data.get(feature_key, {})
"""


def getParas(modelName, targetProperty, combo):
    base_dir = os.path.dirname(__file__)
    jsonPath = os.path.join(base_dir, f'best_{modelName}_for_{targetProperty}_params.json')
    if not os.path.exists(jsonPath):
        return {}

    with open(jsonPath, 'r') as f:
        data = json.load(f)

    #extract only the needed data for this **combo** of features e.g. Density+PV only
    if isinstance(combo, tuple):
        feature_key = ",".join(combo)
    else:
        feature_key = ",".join(combo.columns)

    return data.get(feature_key, {})

def saveParas(modelName, bestPara, targetProperty):
    with open(f'best_{modelName}_for_{targetProperty}_params.json', 'w') as f:
        json.dump(bestPara, f) 

if '__name__' == '__main__':
    modelName = 'SK_GB'
    targetProperty = 'UV'
    features = 'Density'