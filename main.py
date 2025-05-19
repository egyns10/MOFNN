# main.py
from preprocess import readCSV, removeDup, cleanData, saveAsCSV, take2Col
from featurise import featuriseArray
from randomForest import doRandomForest
from linearReg import doLinearReg
from gradBoost import doGradBoost
#from trainTestData import splitData
#from ttd import splitData2

#------------------------------------

#csv_file = 'mof5.csv'
csv_file = 'mof_crystallographic_properties.csv'
readFile = readCSV(csv_file)
print(readFile[:2])

dataNoDup = removeDup(readFile)
print(dataNoDup[:2])

dataCleaned = cleanData(dataNoDup)
print(dataCleaned[:2])

#outputPath = input("Enter the path to save the new csv file to: ")
#saveAsCSV(step2, outputPath)

'''
featurisedData = featuriseArray(dataCleaned)
print("\nfeaturised Data:")
print(featurisedData)
'''

colX = input("Enter the column index for the x axis: ")
colY = input("Enter the column index for the y axis: ")
#X_train, y_train, X_test, y_test = splitData(100, 100, feature_columns=feature_columns)

#header, X_train, y_train, X_test, y_test = splitData2(featurisedData)

dataIsolated = take2Col(dataCleaned[:2] ,colX, colY)
#print("Done")
#print(dataIsolated[:2])

rf_mse, rf_r2 = doRandomForest(dataIsolated)
print(f"\nRandom Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

gb_mse, gb_r2 = doGradBoost(dataIsolated)
print(f"\nGradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

print("\nLinear Regression and Visualization:")
doLinearReg(dataIsolated)
