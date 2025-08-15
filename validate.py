#validate.py
from preprocess import saveAsCSV, isolateCols
import pandas as pd

def saveCleanData(data,filepath):
        csvYN = input(f"Do you want to save {filepath} as a cleaned data? Y/N ")
        if csvYN == "Y" :
            outputPath = filepath.replace(".csv","_cleaned.csv")
            saveAsCSV(data, outputPath)

def columnChoose(data):
    #data is cleaned data including any target value columns
    lengthData = data.shape[1]
    colNValid = False
    max = -1
    print(*data.columns[0:7], sep=", ")
    while max>lengthData or max<0:
        try:
            max = int(input("How many properties would you like to use? "))
        except ValueError:
            continue
        if 0<max<=lengthData:
            chosenProp = []
            while len(chosenProp) != max:
                property = input("What property do you want to use? (Please use exact name as used in the given csv file.) ")
                if property not in list(data.columns)[0:7]:
                    print(f"'{property}' is not a valid column name. Try again.\n")
                    continue
                chosenProp.append(property)
            return chosenProp
        else :
            print("Your value is invalid. Make sure it is in range.\n")
            continue

def UGorUV(propClean):
    print(propClean.columns)
    for i,col in enumerate(propClean.columns):
        print(f'{i}, {col}')

    while True:
        choice = input("UG or UV? ").strip().upper()
        if choice in ["UG",'UV']:
            break
        print(f"'{choice}' is not a valid input.")

    while True:
        userColNo = input(f'What column number are the {choice} values held within the cleaned training data (zero index): ')
        try:
            colNumber = int(userColNo)
        except ValueError:
            print('Invalid input. Try again')
            continue
        if 0 <= colNumber < propClean.shape[1]:
            return isolateCols(propClean,colNumber,"null")
        else:
            print(f"Entered value '{colNumber} is out of bounds.")

if __name__ == "__main__":
    # csvValidate("","x.csv")
    pass
    # UGorUV(pd.read_csv('/Users/nso/Desktop/New MOFs/ASR_Altered.csv'))
