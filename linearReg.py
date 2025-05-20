#linearReg.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def doLinearReg(data, xTitle, yTitle):
    #features are stored in X
    #targets stored in y

    #changes the pandas df into numpy array

    X = data.iloc[:,0].values.reshape(-1, 1)
    y = data.iloc[:,1].values

    #train and tests sets made
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #train the model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    #run the algo and find the error
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

        # Plot true values vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title(xTitle,' vs ',yTitle)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: True vs Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()

    return mse, r2