#random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def doRandomForest(trainData, trainTarget, testData, testTarget, **rfParams):
    X_train = trainData.to_numpy()
    y_train = trainTarget.iloc[:, 0].to_numpy().ravel()
    X_test = testData.to_numpy()
    y_test = testTarget.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    reg = RandomForestRegressor(random_state=42, **rfParams)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def randomTreeXGBoost(trainData, trainTarget, testData, testTarget, **XGrfPara):
    defaultPara = {
        'n_estimators': 1,
        'learning_rate': 1.0,
        'max_depth': 6,
        'colsample_bytree': 0.5,
        'subsample': 0.5,
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    #merge default with parsed parameters if there is any
    parameters = {**defaultPara, **XGrfPara}

    X_train = trainData.to_numpy()
    y_train = trainTarget.iloc[:, 0].to_numpy().ravel()
    X_test = testData.to_numpy()
    y_test = testTarget.iloc[:, 0].to_numpy().ravel()
    
    model = xgb.XGBRegressor(**parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
