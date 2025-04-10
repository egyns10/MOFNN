import numpy as np

def splitData(data, trainingSize, testingSize, feature_columns=None):
    header = data[0]
    
    trainData = data[:trainingSize]  #first `train_size` rows for training
    testData = data[trainingSize:trainingSize+testingSize]   #remaining rows for testing
    #this bit defines the size of the test and training data - originally states at the calling of the function
    #can definitely change it so it's hardcoded to be [:100][100:200] with no call at the top.

    if feature_columns is None:
        #use all columns except the last column as features
        feature_columns = list(range(data.shape[1] - 1))

    X_train = trainData[:, feature_columns]
    y_train = trainData[:, -1]
    X_test = testData[:, feature_columns]
    y_test = testData[:, -1]

    return header, X_train, y_train, X_test, y_test
