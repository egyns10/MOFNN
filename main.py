# main.py
from preprocess import readCSV, removeDup, cleanData, saveAsCSV
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

step1 = removeDup(readFile)
print(step1[:2])

step2 = cleanData(step1)
print(step2[:2])

#outputPath = input("Enter the path to save the new csv file to: ")
#saveAsCSV(step2, outputPath)

'''
featurisedData = featuriseArray(clean)
print("\nfeaturised Data:")
print(featurisedData)

feature_columns = [1, 2]
#X_train, y_train, X_test, y_test = splitData(100, 100, feature_columns=feature_columns)

#header, X_train, y_train, X_test, y_test = splitData2(featurisedData)

rf_mse, rf_r2 = doRandomForest(featurisedData)
print(f"\nRandom Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

gb_mse, gb_r2 = doGradBoost(featurisedData)
print(f"\nGradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

print("\nLinear Regression and Visualization:")
doLinearReg(featurisedData)
'''
