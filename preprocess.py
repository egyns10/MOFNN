# preprocess.py
import csv
import pandas as pd

def readCSV(filepath):
    pd.read_csv(filepath)

def removeDuplicatesFromColumns(data):
#removes duplicate strings within the same column in a 2D array
#note that it transposes data to shift cols into rows
    
    if not data or len(data) < 2:
        return data
    #if there is no data held within the parsed variable or the data is shorter than 2 rows (headers only), return the data as is.

    #rows -> cols
    transposed = list(zip(*data))

    filteredColumns = [column for column in transposed if len(set(column)) > 12]
        #this bit { > 6 } keeps the col if there is more than eight different value in the col (excluding headers)
        #sets cannot have duplicate values
        #this is set to 12 since the training data has 12 Database acronyms
        #everything else of interest has more than 12 unique values

    #cols -> rows
    updated_data = [list(row) for row in zip(*filteredColumns)] if filteredColumns else []

    return updated_data

#BUG
def cleanData(array):
    #removes all strings and null values
    if not array or len(array) < 2:
        return array  
    #if the array fed in is only a header row - only 1 row long

    #find strings in col
    numCols = len(array[0])
    colRM = set()

    for row in array[1:]:
        for colIndex in range(numCols):
            if isinstance(row[colIndex], str):
                colRM.add(colIndex)

    #rm identified cols
    arrayFiltered = [
        [item for idx, item in enumerate(row) if idx not in colRM]
        for row in array
    ]

    #rm null
    arrayClean =[
        [item for item in row if item is not None]
        for row in arrayFiltered
    ]
    return arrayClean

def debugCSVFromArray(data, filePath):
    try:
        with open(filePath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"CSV file successfully created at: {filePath}")
    except Exception as e:
        print(f"An error occurred while creating the CSV file: {e}")