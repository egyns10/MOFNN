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
summary_results.to_csv('/Users/nso/Desktop/summary_results.csv', index=False)
print("Summary saved")
'''

#setup for easy data collection
#TODO: Change this if the calculated accuracy values or the no. of algorithms are altered!
namesML = ['SK_RF','XG_RF','SK_GB','SK_LR']
dfNamesML = pd.DataFrame(data=namesML)
namesAccuracy = ['MSE','R²']
dfNamesAccuracy = pd.DataFrame(data=namesAccuracy)
collectedData, propertyStr = createGrid(features,dfNamesML, dfNamesAccuracy)
#print(collectedData)
#this sets up a blank pd.df to input the r^2 and mse values fpr data collection

filteredData = filterCol(features, propertiesIsolated)
print(filteredData[:2])


rf_mse, rf_r2 = doRandomForest(filteredData, trueValue)
print(f"\nScikit-learn | Random Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")


rfxg_mse, rfxg_r2 = randomTreeXGBoost(filteredData, trueValue)
print(f"\nXGBoost | Random Forest Regressor - MSE: {rfxg_mse:.4f}, R²: {rfxg_r2:.4f}") 


gb_mse, gb_r2 = doGradBoost(filteredData, trueValue)
print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")


lr_mse, lr_r2 = doLinearReg(filteredData, trueValue, propertyStr)
print(f"\nScikit-learn | Linear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")

collectedData.iat[1,1] = rf_mse
collectedData.iat[1,2] = rf_r2
collectedData.iat[2,1] = rfxg_mse
collectedData.iat[2,2] = rfxg_r2
collectedData.iat[3,1] = gb_mse
collectedData.iat[3,2] = gb_r2
collectedData.iat[4,1] = lr_mse
collectedData.iat[4,2] = lr_r2

saveAsCSV(collectedData,f'/Users/nso/Desktop/{propertyStr}.csv')
'''