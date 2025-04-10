# main.py
from preprocess import processCSVFile, removeDuplicatesFromColumns, cleanData
from processCSV import load_csv_to_array, readCSVinNP
from featurise import featuriseArray
from randomForest import doRandomForest
from linearReg import doLinearReg
from gradBoost import doGradBoost
from trainTestData import splitData
#from ttd import splitData2

'''
#filePath = input("Enter the path to the CSV file:\n")
filePath = "/Users/nso/Local Items/Uni/Y3/BEng Project/csv files/mof_crystallographic_properties.csv"
result = processCSVFile(filePath)
print("Processed CSV Data:")
for row in result[:5]:
    print(row)

result = removeDuplicatesFromColumns(result)
print("\nData after removing duplicates from columns:\n")
for row in result[:5]:
    print(row)

clean = cleanData(result)
print("\nRemoved all strings and null spaces:")
for row in result[:5]:
    print(row)

#setupTrainingData(clean)

#outputPath = input("Enter the path to save the new csv file to: ")
#debugCSVFromArray(result, outputPath)


csv_file = 'mof5.csv'
clean = load_csv_to_array(csv_file)
print("Original 2D Array:")
print(clean)
'''
csv_file = 'mof5.csv'
clean = readCSVinNP(csv_file)

featurisedData = featuriseArray(clean)
print("\nfeaturised Data:")
print(featurisedData)

feature_columns = [1, 2]
header, X_train, y_train, X_test, y_test = splitData(featurisedData, 100, 100, feature_columns=feature_columns)

#header, X_train, y_train, X_test, y_test = splitData2(featurisedData)

rf_mse, rf_r2 = doRandomForest((header, X_train, y_train, X_test, y_test))
print(f"\nRandom Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

gb_mse, gb_r2 = doGradBoost((header, X_train, y_train, X_test, y_test))
print(f"\nGradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

print("\nLinear Regression and Visualization:")
doLinearReg((header, X_train, y_train, X_test, y_test))
