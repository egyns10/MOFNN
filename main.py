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
collectedData = createGrid(features,dfNamesML, dfNamesAccuracy)
#print(collectedData)


'''





#grid = [['' for _ in range(lengthData+1)] for _ in range(lengthData+1)]
#gridHeaders = ["Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"]
#print(grid)

gb_mse, gb_r2 = doGradBoost(propertiesClean.iloc[:,[1,2]],gcmcUGIsolated)
#gb_mse, gb_r2 = doGradBoost(propertiesClean[1],propertiesClean[2],gcmcUG)
print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")


dataIsolated = isolateCols(propertiesClean ,features)

rf_mse, rf_r2 = doRandomForest(dataIsolated, trueValue)
print(f"\nScikit-learn | Random Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

SKRFMSE = rf_mse
SKRFR2 = rf_r2


rfxg_mse, rfxg_r2 = randomTreeXGBoost(dataIsolated, trueValue)
print(f"\nXGBoost | Random Forest Regressor - MSE: {rfxg_mse:.4f}, R²: {rfxg_r2:.4f}") 

XGRFMSE[i][j] = rfxg_mse
XGRFR2[i][j] = rfxg_r2

gb_mse, gb_r2 = doGradBoost(dataIsolated, trueValue)
print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

SKBGMSE[i][j] = gb_mse
SKBGR2[i][j] = gb_r2

lr_mse, lr_r2 = doLinearReg(dataIsolated, trueValue, gridHeaders[i], gridHeaders[j])
print(f"\nScikit-learn | Linear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")
print("\nScikit-learn | Linear Regression and Visualization:")

SKLRMSE[i][j] = lr_mse
SKLRR2[i][j] = lr_r2

print(f"Done\nCurrent i: {i} and Current j: {j}")


#df1 = pd.DataFrame(SKRFMSE)
#saveAsCSV(df1, '/Users/nso/Desktop/SKRFMSE.csv')
'''