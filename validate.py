#validate.py
from preprocess import saveAsCSV
import pandas as pd

def csvValidate(data):
    csvYN = "placeholder"
    csvYNValid = False
    while csvYN != "Y" or csvYN != "N":
        csvYN = input("Do you want to save the cleaned data as a CSV? Y/N\n")
        if csvYN == "Y" :
            csvYN = True
            outputPath = input("Enter the path to save the new csv file to: ")
            print("\n")
            saveAsCSV(data, outputPath)
            break
        if csvYN == "N":
            break
        if csvYN != "Y" or csvYN != "N":
            print("Invalid value, try again\n")

def columnChoose(data):
    lengthData = data.shape[1]
    colNValid = False

    while colNValid == False:
        max = int(input("How many properties would you like to use? "))
        if 0<max<=lengthData :
            colNValid = True
            array = pd.DataFrame([[None]*max], columns=range(max))
            print(array)

            for i in range(0,max) :
                property = input("What property do you want to use? (Please use exact name as used in the given csv file.) ")
                if property not in data.columns:
                    print(f"'{property}' is not a valid column name. Try again.\n")
                    return columnChoose(data)  #restart
                array.at[0, i] = property
            return array, max
        
        else :
            print("Your value is invalid. Make sure it is in range.\n")
            return columnChoose(data)  #restart