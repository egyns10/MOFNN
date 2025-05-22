#getData.py
import pandas as pd

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

def intoArray(array,i,j,value):     
    #if arrayMSE[i+1,j+1] != "x":
    array[:,(i+1,j+1)] = value
    return array