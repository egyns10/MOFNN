#gradBoost.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def doGradBoost(trainData, trainTarget, testData, testTarget,**gbParams):
    #features are stored in X
    #targets stored in y
    #changes the pandas df into numpy array
    X_train = trainData.to_numpy()
    y_train = trainTarget.iloc[:, 0].to_numpy().ravel()
    X_test = testData.to_numpy()
    y_test = testTarget.iloc[:, 0].to_numpy().ravel()

    # Now pass them into the model
    gbr = GradientBoostingRegressor(**gbParams, random_state=42)
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
