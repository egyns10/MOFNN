# main.py
from processCSV import load_csv_to_array
from featurise import featuriseArray
from randomForest import doRandomForest
from linearReg import doLinearReg
from gradBoost import doGradBoost
#from trainTestData import splitData
from ttd import splitData

def main():
    csv_file = 'mof5.csv'
    data = load_csv_to_array(csv_file)
    print("Original 2D Array:")
    print(data)

    featurisedData = featuriseArray(data)
    print("\nfeaturised Data:")
    print(featurisedData)

    #feature_columns = [1, 2]
    #header, X_train, y_train, X_test, y_test = splitData(featurisedData, 100, 100, feature_columns=feature_columns)

    header, X_train, y_train, X_test, y_test = splitData(featurisedData)

    rf_mse, rf_r2 = doRandomForest((header, X_train, y_train, X_test, y_test))
    print(f"\nRandom Forest Regressor - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

    gb_mse, gb_r2 = doGradBoost((header, X_train, y_train, X_test, y_test))
    print(f"\nGradient Boosting - MSE: {gb_mse:.4f}, R²: {gb_r2:.4f}")

    print("\nLinear Regression and Visualization:")
    doLinearReg((header, X_train, y_train, X_test, y_test))

#MAIN
#runs main code
if __name__ == '__main__':
    main()
