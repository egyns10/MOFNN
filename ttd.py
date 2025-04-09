import numpy as np

def splitData(data, train_size=100):
    # Extract header
    header = data[0]
    
    # Exclude the header row
    data_no_header = np.array(data[1:])  # Convert to NumPy array (if not already)
    
    # Define split
    train_data = data_no_header[:train_size]  # First `train_size` rows for training
    test_data = data_no_header[train_size:]   # Remaining rows for testing

    # Assuming the last column is the target (y) and the rest are features (X)
    X_train = train_data[:, :-1]  # All columns except the last
    y_train = train_data[:, -1]   # Only the last column

    X_test = test_data[:, :-1]  # All columns except the last
    y_test = test_data[:, -1]   # Only the last column

    return header, X_train, y_train, X_test, y_test
