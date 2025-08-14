#main.py
import pandas as pd
from itertools import combinations
from tqdm import tqdm

from dataSetUp import getTrainingFile,  setUpProp
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import createGrid, filterCol, getParas, saveParas
from validate import columnChoose, UGorUV
from hyperparameters import optimiseRF, optimiseGB, optimiseXGrf

#------------------------------------

#read in all properties from file

userMultiFile = input('Do you want to use two files: one to test and the other to train? Y/N ')

#check to see if training data is default provided data
#training data is mandatory.
filepathTrain = getTrainingFile()

#default training data is held in 'h2_capacity_gcmc.csv'
#this is reflected in the function 

if userMultiFile == 'Y':
    filepathTest = input('Enter the filepath for the testing data: ')
    trainFile = setUpProp(filepathTrain)
    testFile = setUpProp(filepathTest)
else:
    print('Using single file mode:')
    trainFile = setUpProp(filepathTrain)
    testFile = trainFile.copy()
#all the above have outputs in pd.df

#choose properties to use + validate
features = columnChoose(trainFile)
#returns the features chosen (as pd.df) and the number of features chosen

#user chooses on UG or UV
targetValues = UGorUV(testFile)

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

# #headers
# #features.columns = features.columns.map(str).str.strip()
# features = features.iloc[0].dropna().astype(str).str.strip().tolist()

#start of the big loop
#added tqdm for progress and sanity checks
for r_index, r in enumerate(range(1, len(features) + 1), start=1):
    for combo in tqdm(list(combinations(features, r)), desc=f"Feature combos of size {r}"):
        try:
            comboID = ','.join(combo)
            print(f"\n-!!!- Running models for features: {comboID} -!!!-")

            #subset basically acts as a contents page for headerNames, linking each name to the column index
            trainSubset = trainFile[list(combo)]
            testSubset = testFile[list(combo)]
            trainTarget = targetValues
            testTarget = targetValues

            #create grid for the algorithm models
            collectedData, propertyStr = createGrid(
                trainSubset,
                pd.DataFrame([f[0] for f in ml_functions]),
                pd.DataFrame(namesAccuracy)
            )

        except KeyError as e:
            print(f"Skipping combo {combo} due to missing column: {e}") #if user requests test for properties that are not present!!!
        except Exception as e:
            print(f"Error running combo {combo}: {e}")

        #run each model
        for modelName, modelFunc, optimiserFunc in ml_functions:
            bestParaSaved = getParas(modelName)

            try:
                if comboID in bestParaSaved:
                    bestParas = bestParaSaved[comboID]
                    print(f"Using saved parameters for {modelName} and combo {comboID}")
                elif optimiserFunc:
                    print(f"Cannot find saved parameters for {modelName} and combo {comboID}")
                    print("Optimising hyperparameters...")
                    bestParas, _ = optimiserFunc(trainFile, targetValues)
                    bestParaSaved[comboID] = bestParas
                    saveParas(modelName, bestParaSaved)
                else:
                    bestParas = {}

                if optimiserFunc:
                    mse, r2 = modelFunc(trainSubset, trainTarget, testSubset, testTarget, **bestParas)
                else:
                    mse, r2 = modelFunc(trainSubset, trainTarget, testSubset, testTarget, propertyStr)  #linear regression

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


#save summary at the end
summary_results.to_csv('/Users/nso/Desktop/summary_results.csv', index=False)
print("Summary saved")
