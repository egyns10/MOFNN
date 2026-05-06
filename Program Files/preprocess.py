#preprocess.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def readCSV(filepath):
    return pd.read_csv(filepath)

def removeDup(data):
    return data.drop_duplicates(keep='first').reset_index(drop=True)

def cleanData(data):
    noString = data.map(lambda x: x if (isinstance(x, (int, float, np.number)) and not pd.isnull(x)) else np.nan)
    return noString.dropna(axis=1, how='all').reset_index(drop=True)

def saveAsCSV(data, filePath):
    data.to_csv(filePath, index=False)

def isolateCols(data, colX, colY):
    if colY == "null" :
        colX = int(colX)
        return data.iloc[:,[colX]]
    else :
        colX = int(colX)
        colY = int(colY)
        start = min(colX, colY)
        end = max(colX, colY) + 1  #+1 since iloc slicing is exclusive on the end
        return data.iloc[:, start:end]

def imputeMissingValues(data):
    """
    Impute missing values in the dataset using mean imputation for numeric columns.
    """
    imputer = SimpleImputer(strategy='mean')
    imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed

def selectMLFunctions():
    available_models = [
        ('SK_RF', 'Scikit-learn Random Forest'),
        ('XG_RF', 'XGBoost Random Forest'),
        ('SK_GB', 'Scikit-learn Gradient Boosting'),
        ('SK_LR', 'Scikit-learn Linear Regression'),
        ('SK_SGD', 'Scikit-learn SGD Regressor')
    ]
    
    print("Available ML models:")
    for i, (code, name) in enumerate(available_models, 1):
        print(f"{i}. {code}: {name}")
    
    selected = []
    while True:
        try:
            choice = input("Enter model numbers to use (comma-separated, or 'all' for all, 'done' to finish): ").strip()
            if choice.lower() == 'done':
                break
            if choice.lower() == 'all':
                selected = [i-1 for i in range(1, len(available_models)+1)]
                break
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(available_models) and idx not in selected:
                    selected.append(idx)
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
    
    # Import here to avoid circular imports
    from randomForest import doRandomForest, randomTreeXGBoost
    from linearReg import doLinearReg, doSGDReg
    from gradBoost import doGradBoost
    from hyperparameters import optimiseRF, optimiseGB, optimiseXGrf
    
    model_map = {
        0: ('SK_RF', doRandomForest, optimiseRF),
        1: ('XG_RF', randomTreeXGBoost, optimiseXGrf),
        2: ('SK_GB', doGradBoost, optimiseGB),
        3: ('SK_LR', doLinearReg, None),
        4: ('SK_SGD', doSGDReg, None)
    }
    
    return [model_map[idx] for idx in selected]
