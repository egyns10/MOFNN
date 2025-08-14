#dataSetUp.py

from preprocess import readCSV, removeDup, cleanData, isolateCols
from validate import csvValidate

def defaultTrain():
    userDefaultTrain = "placeholder"
    while userDefaultTrain != 'Y' or defaultTrain != 'N':
        defaultTrain = input('Use default training data? Y/N')
        if userDefaultTrain == 'Y':
            filepathTrain = 'h2_capacity_gcmc.csv'
        elif userDefaultTrain == 'N':
            filepathTrain = input('Enter the Training Data filepath: ')
    return userDefaultTrain, filepathTrain

def setUpProp(filepath):
    propertiesReadFile = readCSV(filepath)
    propertiesNoDup = removeDup(propertiesReadFile)
    propertiesClean = cleanData(propertiesNoDup)
    propertiesIsolated = isolateCols(propertiesClean,0,6) #takes out only the properties and nothing else - hard coded.
    propertiesIsolated.columns = propertiesIsolated.columns.astype(str).str.strip() #removes any lingering whitespaces
    csvValidate(propertiesIsolated)
    print(propertiesIsolated[:2])
    return propertiesIsolated
