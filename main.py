#main.py
from preprocess import readCSV, removeDup, cleanData, saveAsCSV, take2Col
from randomForest import doRandomForest, randomTreeXGBoost
from linearReg import doLinearReg
from gradBoost import doGradBoost
from getData import intoArray

#------------------------------------

#csv_file = 'mof5.csv'
csv_file = 'mof_crystallographic_properties.csv'
readFile = readCSV(csv_file)

dataNoDup = removeDup(readFile)

dataCleaned = cleanData(dataNoDup)

'''
csvYN = "placeholder"
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
'''


'''
colX = -1
colY = -1
lengthData = dataCleaned.shape[1]
print(lengthData)
colNValid = False
while colNValid == False:
    colX = int(input("Enter the column index for the x axis: "))
    colY = int(input("Enter the column index for the y axis: "))
    if (colX in range(0,lengthData-1)) and (colY in range(0,lengthData-1)) and (colY != colX):
        colNValid = True
        break
    if colNValid == False:
        print("Your values are invalid. Make sure they are in range.\n")
'''

colX = [0,1,2,3,4,5,6,7]
colY = [0,1,2,3,4,5,6,7]

SKRFMSE = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
SKRFR2 = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
XGRFMSE = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
XGRFR2 = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
SKBGMSE = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
SKBGR2 = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
SKLRMSE = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]
SKLRR2 = [["","Density","GSA","VSA","Void Fraction","Pore Volume","Largest Cavity Diameter","Pore Limiting Diameter"],["Density","x","","","","","",""],["GSA","","x","","","","",""],["VSA","","","x","","","",""], ["Void Fraction","","","","x","","",""],["Pore Volume","","","","","x","",""],["Largest Cavity Diameter","","","","","","x",""],["Pore Limiting Diameter","","","","","","","x"]]

headers = dataCleaned.columns.toList()

for i in colX :
    for j in colY:

        dataIsolated = take2Col(dataCleaned ,colX[i], colY[j])

        rf_mse, rf_r2 = doRandomForest(dataIsolated)
        print(f"\nScikit-learn | Random Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

        SKRFMSE = intoArray(SKRFMSE,i,j,rf_mse)
        SKRFR2 = intoArray(SKRFR2,i,j,rf_r2)


        rfxg_mse, rfxg_r2 = randomTreeXGBoost(dataIsolated)
        print(f"\nXGBoost | Random Forest Regressor - MSE: {rfxg_mse:.4f}, R²: {rfxg_r2:.4f}") 

        XGRFMSE = intoArray(XGRFMSE,i,j,rfxg_mse)
        XGRFR2 = intoArray(XGRFR2,i,j,rfxg_r2)

        gb_mse, gb_r2 = doGradBoost(dataIsolated)
        print(f"\nScikit-learn | Gradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

        SKBGMSE = intoArray(SKBGMSE,i,j,gb_mse)
        SKBGR2 = intoArray(SKBGR2,i,j,gb_r2)

        lr_mse, lr_r2 = doLinearReg(dataIsolated,headers[colX],headers[colY])
        print(f"\nScikit-learn | Linear Regression = MSE {lr_mse:.4f},R²: {lr_r2:.4f}")
        print("\nScikit-learn | Linear Regression and Visualization:")

        SKLRMSE = intoArray(SKLRMSE,i,j,lr_mse)
        SKLRR2 = intoArray(SKLRR2,i,j,lr_r2)

saveAsCSV(SKRFMSE, '/Users/nso/Desktop/SKRFMSE.csv')
saveAsCSV(SKRFR2, '/Users/nso/Desktop/SKRFR2.csv')

saveAsCSV(XGRFMSE, '/Users/nso/Desktop/XGRFMSE.csv')
saveAsCSV(XGRFR2, '/Users/nso/Desktop/XGRFR2.csv')

saveAsCSV(SKBGMSE, '/Users/nso/Desktop/SKBGMSE.csv')
saveAsCSV(SKBGR2, '/Users/nso/Desktop/SKBGR2.csv')

saveAsCSV(SKLRMSE, '/Users/nso/Desktop/SKLRMSE.csv')
saveAsCSV(SKLRR2, '/Users/nso/Desktop/SKLRR2.csv')
