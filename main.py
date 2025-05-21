#main.py
import pandas as pd
import numpy as np

from preprocess import readCSV, removeDup, cleanData, saveAsCSV, isolateCols#, take2Col, extractColumn
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import createGrid, intoArray
from validate import csvValidate, columnChoose

#------------------------------------

filepath = 'h2_capacity_gcmc.csv'
propertiesReadFile = readCSV(filepath)
propertiesNoDup = removeDup(propertiesReadFile)
propertiesClean = cleanData(propertiesNoDup)
propertiesIsolated = isolateCols(propertiesClean,0,6)
#save as csv? + validate
csvValidate(propertiesClean)
print(propertiesIsolated[:2])

gcmcReadFile = readCSV(filepath)
gcmcNoDup = removeDup(gcmcReadFile)
gcmcClean = cleanData(gcmcNoDup)
gcmcUGIsolated = isolateCols(gcmcClean,7,"null")
gcmcUVIsolated = isolateCols(gcmcClean,8,"null")

print(gcmcUGIsolated[:2])
#gcmcUG = extractColumn(gcmcClean,'UG at PS ')
#gcmcUV = extractColumn(gcmcClean,'UV at PS ')


gb_mse, gb_r2 = doGradBoost(propertiesClean.iloc[:,[1,2]],gcmcUGIsolated)
#gb_mse, gb_r2 = doGradBoost(propertiesClean[1],propertiesClean[2],gcmcUG)
print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

'''
#choose properties to use + validate
columnChoose(propertiesClean)

lengthData = propertiesClean.shape[1]
grid = [['' for _ in range(lengthData+1)] for _ in range(lengthData+1)]
gridHeaders = ["Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"]

SKRFMSE = createGrid(gridHeaders)
SKRFR2  = createGrid(gridHeaders)
XGRFMSE = createGrid(gridHeaders)
XGRFR2 = createGrid(gridHeaders)
SKBGMSE = createGrid(gridHeaders)
SKBGR2 = createGrid(gridHeaders)
SKLRMSE = createGrid(gridHeaders)
SKLRR2 = createGrid(gridHeaders)


for i in range(1,lengthData+1):
    for j in range(1,lengthData+1):

        dataIsolated = take2Col(dataCleaned ,i-1, j-1)

        rf_mse, rf_r2 = doRandomForest(dataIsolated)
        print(f"\nScikit-learn | Random Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

        SKRFMSE[i][j] = rf_mse
        SKRFR2[i][j] = rf_r2


        rfxg_mse, rfxg_r2 = randomTreeXGBoost(dataIsolated)
        print(f"\nXGBoost | Random Forest Regressor - MSE: {rfxg_mse:.4f}, R²: {rfxg_r2:.4f}") 

        XGRFMSE[i][j] = rfxg_mse
        XGRFR2[i][j] = rfxg_r2

        gb_mse, gb_r2 = doGradBoost(dataIsolated)
        print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

        SKBGMSE[i][j] = gb_mse
        SKBGR2[i][j] = gb_r2

        lr_mse, lr_r2 = doLinearReg(dataIsolated,gridHeaders[i],gridHeaders[j])
        print(f"\nScikit-learn | Linear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")
        print("\nScikit-learn | Linear Regression and Visualization:")

        SKLRMSE[i][j] = lr_mse
        SKLRR2[i][j] = lr_r2

        print(f"Done\nCurrent i: {i} and Current j: {j}")

df1 = pd.DataFrame(SKRFMSE)
saveAsCSV(df1, '/Users/nso/Desktop/SKRFMSE.csv')

df2 = pd.DataFrame(SKRFR2)
saveAsCSV(df2, '/Users/nso/Desktop/SKRFR2.csv')

df3 = pd.DataFrame(XGRFMSE)
saveAsCSV(df3, '/Users/nso/Desktop/XGRFMSE.csv')

df4 = pd.DataFrame(XGRFR2)
saveAsCSV(df4, '/Users/nso/Desktop/XGRFR2.csv')

df5 = pd.DataFrame(SKBGMSE)
saveAsCSV(df5, '/Users/nso/Desktop/SKBGMSE.csv')

df6 = pd.DataFrame(SKBGR2)
saveAsCSV(df6, '/Users/nso/Desktop/SKBGR2.csv')

df7 = pd.DataFrame(SKLRMSE)
saveAsCSV(df7, '/Users/nso/Desktop/SKLRMSE.csv')

df8 = pd.DataFrame(SKLRR2)
saveAsCSV(df8, '/Users/nso/Desktop/SKLRR2.csv')
'''