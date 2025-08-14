#linearReg.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def doLinearReg(trainData, trainTarget, testData, testTarget):
    #features are stored in X
    #targets stored in y

    #changes the pandas df into numpy array

    X_train = trainData.to_numpy()
    y_train = trainTarget.iloc[:, 0].to_numpy().ravel()
    X_test = testData.to_numpy()
    y_test = testTarget.iloc[:, 0].to_numpy().ravel()

    #train the model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    #run the algo and find the error
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    '''
    #plot true values vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title("Predicted vs True Values")
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid()
    #plt.savefig(f"/Users/nso/Desktop/BEng/{xTitle}VS{yTitle}")
    plt.savefig(f"/Users/nso/Desktop/BEng2/{properties}.jpg")
    print("Figure saved")
    '''

    return mse, r2