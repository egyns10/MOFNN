#main.py
import pandas as pd
import numpy as np

from preprocess import readCSV, removeDup, cleanData, saveAsCSV, isolateCols
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import createGrid, intoArray
from validate import csvValidate, columnChoose, UGorUV

#------------------------------------

#read in all properties from file
filepath = 'h2_capacity_gcmc.csv'
propertiesReadFile = readCSV(filepath)
propertiesNoDup = removeDup(propertiesReadFile)
propertiesClean = cleanData(propertiesNoDup)
propertiesIsolated = isolateCols(propertiesClean,0,6)
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


#choose properties to use + validate
features, max = columnChoose(propertiesClean)
#returns the features chosen (as pd.df) and the number of features chosen

#user chooses on UG or UV
trueValue = UGorUV(gcmcUGIsolated,gcmcUVIsolated)

#setup for easy data collection
#TODO: Change this if the calculated accuracy values or the no. of algorithms are altered!
namesML = ['SK_RF','XG_RF','SK_GB','SK_LR']
dfNamesML = pd.DataFrame(data=namesML)
namesAccuracy = ['MSE','R²']
dfNamesAccuracy = pd.DataFrame(data=namesAccuracy)
collectedData, propertyStr = createGrid(features,dfNamesML, dfNamesAccuracy)
#print(collectedData)


#grid = [['' for _ in range(lengthData+1)] for _ in range(lengthData+1)]
#gridHeaders = ["Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"]
#print(grid)

#THIS IS AN EXAMPLE - not done properly since the [1,2] is hard coded in.
gb_mse, gb_r2 = doGradBoost(propertiesClean.iloc[:,[1,2]],gcmcUGIsolated)
#gb_mse, gb_r2 = doGradBoost(propertiesClean[1],propertiesClean[2],gcmcUG)

#TODO: change algorithm call to include multiple properties!!!! - see above for example
dataIsolated = isolateCols(propertiesClean ,features)

rf_mse, rf_r2 = doRandomForest(dataIsolated, trueValue)
print(f"\nScikit-learn | Random Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

intoArray(collectedData,1,1,rf_mse)


rfxg_mse, rfxg_r2 = randomTreeXGBoost(dataIsolated, trueValue)
print(f"\nXGBoost | Random Forest Regressor - MSE: {rfxg_mse:.4f}, R²: {rfxg_r2:.4f}") 


gb_mse, gb_r2 = doGradBoost(dataIsolated, trueValue)
print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")


#lr_mse, lr_r2 = doLinearReg(dataIsolated, trueValue, gridHeaders[i], gridHeaders[j])
lr_mse, lr_r2 = doLinearReg(dataIsolated, trueValue, propertyStr)
print(f"\nScikit-learn | Linear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")
print("\nScikit-learn | Linear Regression and Visualization:")

collectedData.iat[1,1] = rf_mse
collectedData.iat[1,2] = rf_r2
collectedData.iat[2,1] = rfxg_mse
collectedData.iat[2,2] = rfxg_r2
collectedData.iat[3,1] = gb_mse
collectedData.iat[3,2] = gb_r2
collectedData.iat[4,1] = lr_mse
collectedData.iat[4,2] = lr_r2

saveAsCSV(collectedData,f'/Users/nso/Desktop/{propertyStr}.csv')