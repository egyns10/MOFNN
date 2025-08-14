#dataSetUp.py

from preprocess import readCSV, removeDup, cleanData
from validate import csvValidate

def getTrainingFile():
    userDefaultTrain = "placeholder"
    while userDefaultTrain != 'Y' and userDefaultTrain != 'N':
        userDefaultTrain = input('Use default training data? Y/N ')
        if userDefaultTrain == 'Y':
            filepathTrain = 'h2_capacity_gcmc.csv'
        elif userDefaultTrain == 'N':
            filepathTrain = input('Enter the Training Data filepath: ')
    return filepathTrain

def setUpProp(filepath):
    propertiesNoDup = dedupedProp(filepath)
    propertiesClean = cleanData(propertiesNoDup)
    propertiesClean.columns = propertiesClean.columns.astype(str).str.strip() #removes any lingering whitespaces
    csvValidate(propertiesClean)
    # print(propertiesClean[:2])
    return propertiesClean

def dedupedProp(filepath):
    propertiesReadFile = readCSV(filepath)
    return removeDup(propertiesReadFile)
