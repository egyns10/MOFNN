#validate.py
from preprocess import saveAsCSV

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
    colX = -1
    colY = -1
    lengthData = data.shape[1]

    colNValid = False
    while colNValid == False:
        colX = int(input("Enter the column index for the x axis: "))
        colY = int(input("Enter the column index for the y axis: "))
        if (colX in range(0,lengthData-1)) and (colY in range(0,lengthData-1)) and (colY != colX):
            colNValid = True
            break
        if colNValid == False:
            print("Your values are invalid. Make sure they are in range.\n")