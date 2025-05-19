#gradBoost.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def doGradBoost(data):
    #features are stored in X
    #targets stored in y
    X = data.iloc[:,0].values.reshape(-1, 1)
    y = data.iloc[:,1].values

    #train and tests sets made
    X_train, X_test, y_train, y_test = train_test_split(X, y, testSize=0.3, randomState=42)

    gbr = GradientBoostingRegressor(nEstimators=50, learningRate=0.075, randomState=42)
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
