# main.py
from preprocess import readCSV, removeDup, cleanData, saveAsCSV, take2Col
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



csvYNValid = False
while csvYN != "Y" or csvYN != "N":
    csvYN = input("Do you want to save the cleaned data as a CSV? Y/N\n")
    if csvYN == "Y" :
        csvYN = True
        outputPath = input("Enter the path to save the new csv file to: ")
        print("\n")
        saveAsCSV(dataCleaned, outputPath)
        break
    if csvYN == "N":
        break
    if csvYN != "Y" or csvYN != "N":
        print("Invalid value, try again\n")

colX = -1
colY = -1
lengthData = dataCleaned.shape[1]
print(lengthData)
colNValid = False
while colNValid == False:
    colX = int(input("Enter the column index for the x axis: "))
    colY = int(input("Enter the column index for the y axis: "))
    if (colX in range(0,lengthData-1)) and (colY in range(0,lengthData-1)) and (colY is not colX):
        colNValid = True
        break
    if colNValid == False:
        print("Your values are invalid. Make sure they are in range.\n")

dataIsolated = take2Col(dataCleaned ,colX, colY)
print("Done\n")
print(dataIsolated[:2])

rf_mse, rf_r2 = doRandomForest(dataIsolated)
print(f"\nRandom Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

gb_mse, gb_r2 = doGradBoost(dataIsolated)
print(f"\nGradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

lr_mse, lr_r2 = doLinearReg(dataIsolated)
print(f"\nLinear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")
print("\nLinear Regression and Visualization:")
