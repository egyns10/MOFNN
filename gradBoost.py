#gradBoost.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def doGradBoost(data,true):
    #features are stored in X
    #targets stored in y

    #changes the pandas df into numpy array
    X = data.to_numpy() 
    y = true.iloc[:, 0].to_numpy().ravel()

    #train and tests sets made
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.075, random_state=42)
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
