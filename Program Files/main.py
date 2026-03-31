#main.py
import pandas as pd
from itertools import combinations
from tqdm import tqdm

from dataSetUp import getTrainingFile,  setUpProp, dedupedProp
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg, doSGDReg
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
    filepathTest = input('Enter the filepath for the testing data: ')
    trainFile = setUpProp(filepathTrain)
    testFile = setUpProp(filepathTest)
else:
    print('Switching to single file mode...')
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
    #('SK_RF', doRandomForest, optimiseRF),
    #('XG_RF', randomTreeXGBoost, optimiseXGrf),
    #('SK_GB', doGradBoost, optimiseGB),
    ('SK_LR', doLinearReg, None),                   #linear regression does not have any hyperparameters to be tuned.
    ('SK_SGD', doSGDReg, None)
]
summaryResults = pd.DataFrame(columns=['Features', 'Model', 'MSE', 'R²'])
mofRecords = []

#start of the big loop
#added tqdm for progress and sanity checks
for r in range(1, len(features) +1):
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

            #TODO: This repeats again below - change so it's modular
            if targetValues.columns[0] == 'UG at PS':
                targetProperty = 'UG'
            elif targetValues.columns[0] == 'UV at PS':
                targetProperty = 'UV'


            
            try:

                bestParas = getParas(modelName, targetProperty, combo)

                if bestParas:
                    print(f"Using saved parameters for {comboID}")
                elif optimiserFunc:
                    print(f"Cannot find saved parameters for {modelName} and combo {comboID}")
                    print("Optimising hyperparameters...")
                    bestParas, _ = optimiserFunc(trainFile, targetValues)
                    saveParas(modelName, {comboID: bestParas}, targetProperty)
                else:
                    bestParas = {}  #here for linear regression models which do not need tuned hyperparameters

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

                namesTestFile = dedupedProp(testFile)
                newMOFs = {
                    i: namesTestFile.at[i, 'coreid']
                    for i in predictedValues
                    if i in namesTestFile.index
                }

                for i in predictedValues:
                    if i in namesTestFile.index:
                        mofRecords.append({
                        'MOF': namesTestFile.at[i, 'coreid'],
                        'PredictedValue': testTarget.at[i, targetValues.columns[0]],
                        'Model': modelName,
                        'Features': comboID
                        })

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
#summary-user_filepath = input("Where would you like to save the summary file to? ")
#summaryResults.to_csv(summary-user_filepath, index=False)
summaryResults.to_csv('/Users/nso/Desktop/summary_results.csv', index=False)

#remove second duplicates then convert list to pd.series and save as csv
MOFsDF = pd.DataFrame(mofRecords).drop_duplicates(subset='MOF')

#MOFs_of_interest-user_filepath = input("Where would you like to save the file MOFs_of_interest to? ")
#MOFsDF.to_csv(MOFs_of_interest-user_filepath, index=False)
MOFsDF.to_csv("/Users/nso/Desktop/MOFs_of_interest.csv", index=False)

print("Summary saved")
