import numpy as np

def splitData(data, trainingSize, testingSize, feature_columns=None):
    # Extract header
    header = data[0]
    
    # Exclude the header row
    cleanData = np.array(data[1:])  # Convert to NumPy array (if not already)
    
    # Define split
    trainData = cleanData[:trainingSize]  # First `train_size` rows for training
    testData = cleanData[trainingSize:trainingSize+testingSize]   # Remaining rows for testing
    #this bit defines the size of the test and training data - originally states at the calling of the function
    #can definitely change it so it's hardcoded to be [:100][100:200] with no call at the top.

    # Determine feature columns
    if feature_columns is None:
        # Use all columns except the last column as features
        feature_columns = list(range(cleanData.shape[1] - 1))

    # Assuming the last column is the target (y)
    X_train = trainData[:, feature_columns]  # Selected feature columns
    y_train = trainData[:, -1]               # Last column as target

    X_test = testData[:, feature_columns]  # Selected feature columns
    y_test = testData[:, -1]               # Last column as target

    return header, X_train, y_train, X_test, y_test
