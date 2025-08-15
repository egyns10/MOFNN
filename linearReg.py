#linearReg.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def doLinearReg(trainData, trainTarget, testData):
    #features are stored in X
    #targets stored in y

    #train
    X = trainData.to_numpy() 
    y = trainTarget.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #train the model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    #run the algo and find the error
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    '''
    #plots true values vs predicted values (all based on training data e.g. how close the algo got)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title("Predicted vs True Values")
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid()
    #plt.savefig(f"{xTitle}VS{yTitle}")
    plt.savefig(f"{properties}.jpg")
    print("Figure saved")
    '''

    #predict
    predictData = testData.to_numpy()
    predictions = reg.predict(predictData)
    #compared to finding y_pred, this predicts UG or UV values given the testing data

    highUG = [i for i, pred in enumerate(predictions) if pred > 35]
    highUV = [i for i, pred in enumerate(predictions) if pred > 38] 

    return mse, r2, highUG, highUV