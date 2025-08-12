#main.py
import pandas as pd
from itertools import combinations
from tqdm import tqdm

from preprocess import readCSV, removeDup, cleanData, saveAsCSV, isolateCols
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import createGrid, filterCol, getParas, saveParas
from validate import csvValidate, columnChoose, UGorUV
from hyperparameters import optimiseRF, optimiseGB, optimiseXGrf

#------------------------------------

#read in all properties from file
filepath = 'h2_capacity_gcmc.csv'
propertiesReadFile = readCSV(filepath)
propertiesNoDup = removeDup(propertiesReadFile)
propertiesClean = cleanData(propertiesNoDup)
propertiesIsolated = isolateCols(propertiesClean,0,6) #takes out only the properties and nothing else - hard coded.
#save as csv? + validate
csvValidate(propertiesClean)
print(propertiesIsolated[:2])

#read in H2 capacities from same file
gcmcReadFile = readCSV(filepath)
gcmcNoDup = removeDup(gcmcReadFile)
gcmcClean = cleanData(gcmcNoDup)
gcmcUGIsolated = isolateCols(gcmcClean,7,"null")
gcmcUVIsolated = isolateCols(gcmcClean,8,"null")
print(gcmcUGIsolated[:2])

#all the above is now in pd.df form


#choose properties to use + validate
features, max = columnChoose(propertiesClean)
#returns the features chosen (as pd.df) and the number of features chosen

#user chooses on UG or UV
trueValue = UGorUV(gcmcUGIsolated,gcmcUVIsolated)

#--------------------------

#initialising set up
ml_functions = [
    ('SK_RF', doRandomForest, optimiseRF),
    ('XG_RF', randomTreeXGBoost, optimiseXGrf),
    ('SK_GB', doGradBoost, optimiseGB),
    ('SK_LR', doLinearReg, None),           #linear regression does not have any hyperparameters to be tuned.
]
namesAccuracy = ['MSE', 'R²']
summary_results = pd.DataFrame(columns=['Features', 'Model', 'MSE', 'R²'])

#headers
features.columns = features.columns.map(str).str.strip()
#headers = list(features.columns)
headers = features.iloc[:, 0].dropna().astype(str).str.strip().tolist()


#start of the big loop
#added tqdm for progress and sanity checks
for r_index, r in enumerate(range(1, len(headers) + 1), start=1):
    for combo in tqdm(list(combinations(headers, r)), desc=f"Feature combos of size {r}"):
        try:
            comboID = ','.join(combo)
            print(f"\n-!!!- Running models for features: {comboID} -!!!-")

            #subset DataFrame to selected features
            featureSubset = features[list(combo)]
            filteredData = filterCol(featureSubset, propertiesIsolated)

            #create grid for the algorithm models
            collectedData, propertyStr = createGrid(
                featureSubset,
                pd.DataFrame([f[0] for f in ml_functions]),
                pd.DataFrame(namesAccuracy)
            )

            #run each model
            for modelName, modelFunc, optimiserFunc in ml_functions:
                try:
                    try:
                        bestParaSaved = getParas(modelName)
                    except FileNotFoundError:
                        bestParaSaved = {}

                    if comboID in bestParaSaved:
                        bestParas = bestParaSaved[comboID]
                        (f"Using saved parameters for {modelName} and combo {comboID}")
                    elif optimiserFunc:
                        print(f"Cannot find saved parameters for {modelName} and combo {comboID}")
                        print("Optimising...")
                        bestParas, _ = optimiserFunc(filteredData, trueValue)
                        bestParaSaved[comboID] = bestParas
                        saveParas(modelName, bestParaSaved)
                    else:
                        bestParas = {}

                    if optimiserFunc:
                        mse, r2 = modelFunc(filteredData, trueValue, **bestParas)
                    else:
                        mse, r2 = modelFunc(filteredData, trueValue, propertyStr)  #linear regression

                    print(f"{modelName} - MSE: {mse:.4f}, R²: {r2:.4f}")

                    summary_results = pd.concat([
                        summary_results,
                        pd.DataFrame([{
                            'Features': comboID,
                            'Model': modelName,
                            'MSE': mse,
                            'R²': r2
                        }])
                    ], ignore_index=True)

                except Exception as e:
                    print(f"Error running model {modelName} for combo {comboID}: {e}")

        except KeyError as e:
            print(f"Skipping combo {combo} due to missing column: {e}") #if user requests test for properties that are not present!!!
        except Exception as e:
            print(f"Error running combo {combo}: {e}")

#save summary at the end
summary_results.to_csv('/Users/nso/Desktop/summary_results.csv', index=False)
print("Summary saved")
