#main.py
import pandas as pd
from itertools import combinations
from tqdm import tqdm

from dataSetUp import getTrainingFile,  setUpProp, dedupedProp
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import getParas, saveParas
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
    # filepathTest = input('Enter the filepath for the testing data: ')
    filepathTest = '/Users/nso/Desktop/New MOFs/ASR_Altered.csv'
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
targetValues = UGorUV(trainFile)

#--------------------------

#initialising set up
ml_functions = [
    ('SK_RF', doRandomForest, optimiseRF),
    ('XG_RF', randomTreeXGBoost, optimiseXGrf),
    ('SK_GB', doGradBoost, optimiseGB),
    ('SK_LR', doLinearReg, None),           #linear regression does not have any hyperparameters to be tuned.
]

summaryResults = pd.DataFrame(columns=['Features', 'Model', 'MSE', 'R²'])
MOFsSeries = pd.Series(dtype='object')  #empty series

# #headers
# #features.columns = features.columns.map(str).str.strip()
# headerName = features.iloc[0].dropna().astype(str).str.strip().tolist()

#start of the big loop
#added tqdm for progress and sanity checks
for r in range(1, len(features) + 1):
    for combo in tqdm(list(combinations(features, r)), desc=f"Feature combos of size {r}"):
        try:
            comboID = ','.join(combo)
            print(f"\n-!!!- Running models for features: {comboID} -!!!-")

            #subset basically acts as a contents page for headerNames, linking each name to the column index
            trainSubset = trainFile[list(combo)]
            testSubset = testFile[list(combo)]
            trainTarget = targetValues
            testTarget = targetValues

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
                    mse, r2, bestUG, bestUV = modelFunc(trainSubset, trainTarget, testSubset, **bestParas)
                else:
                    mse, r2, bestUG, bestUV = modelFunc(trainSubset, trainTarget, testSubset)  #linear regression
                #bestUG and bestUV are a list of indices of which link MOFs of interest
                print(f"{modelName} - MSE: {mse:.4f}, R²: {r2:.4f}")

                if targetValues.columns[0] == 'UG at PS':
                    predictedValues = bestUG
                elif targetValues.columns[0] == 'UV at PS':
                    predictedValues = bestUV
                else:
                    predictedValues = []
                    print('Could not find MOF names.')

                namesTestFile = dedupedProp(filepathTest)
                #MOFsNames = namesTestFile.columns[0]
                newMOFs = {
                    i: namesTestFile.at[i, 'coreid']
                    for i in predictedValues
                    if i in namesTestFile.index
                }
                MOFsSeries = pd.concat([MOFsSeries, pd.Series(newMOFs)], axis=0)


                summaryResults = pd.concat([
                    summaryResults,
                    pd.DataFrame([{
                        'Features': comboID,
                        'Model': modelName,
                        'MSE': mse,
                        'R²': r2
                        }])
                ], ignore_index=True)

            except Exception as e:
                print(f"Error running model {modelName} for combo {comboID}: {e}")
                raise e

#save summary at the end
summaryResults.to_csv('/Users/nso/Desktop/summary_results.csv', index=False)

#remove second duplicates then convert list to pd.series and save as csv
MOFsSeries = MOFsSeries[~MOFsSeries.index.duplicated(keep='first')]
#set 'column' name (as series, only 1 col)
MOFsSeries.name = 'MOF'
MOFsSeries.to_csv("/Users/nso/Desktop/MOFs_of_interest.csv")

print("Summary saved")
