#main.py
import pandas as pd

from preprocess import readCSV, removeDup, cleanData, saveAsCSV, isolateCols
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import createGrid, filterCol
from validate import csvValidate, columnChoose, UGorUV
from itertools import combinations

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


#-----------------------------------


# Set up models and results logging
ml_functions = [
    ('SK_RF', doRandomForest),
    ('XG_RF', randomTreeXGBoost),
    ('SK_GB', doGradBoost),
    ('SK_LR', doLinearReg),
]
namesAccuracy = ['MSE', 'R²']
summary_results = pd.DataFrame(columns=['Features', 'Model', 'MSE', 'R²'])

# Clean headers
features.columns = features.columns.map(str).str.strip()
headers = list(features.columns)

# Loop through all combinations from 1 to len(headers)
for r in range(1, len(headers) + 1):
    for combo in combinations(headers, r):
        try:
            # Subset DataFrame to selected features
            featureSubset = features[list(combo)]
            filteredData = filterCol(featureSubset, propertiesIsolated)

            # Create model grid
            collectedData, propertyStr = createGrid(
                featureSubset,
                pd.DataFrame([f[0] for f in ml_functions]),
                pd.DataFrame(namesAccuracy)
            )

            print(f"\n--- Running models for features: {', '.join(combo)} ---")

            # Run each model
            for modelName, modelFunc in ml_functions:
                if modelName == 'SK_LR':
                    mse, r2 = modelFunc(filteredData, trueValue, propertyStr)
                else:
                    mse, r2 = modelFunc(filteredData, trueValue)

                print(f"{modelName} - MSE: {mse:.4f}, R²: {r2:.4f}")

                summary_results = pd.concat([
                    summary_results,
                    pd.DataFrame([{
                        'Features': ', '.join(combo),
                        'Model': modelName,
                        'MSE': mse,
                        'R²': r2
                    }])
                ], ignore_index=True)

        except KeyError as e:
            print(f"Skipping combo {combo} due to missing column: {e}")
        except Exception as e:
            print(f"Error running combo {combo}: {e}")

# Save summary at the end
summary_results.to_csv('/Users/nso/Desktop/summary_results_bare.csv', index=False)
print("Summary saved")
