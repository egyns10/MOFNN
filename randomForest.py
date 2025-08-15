#random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb

def doRandomForest(trainData, trainTarget, testData, **rfParams):

    #below is all the training process which generates mse and R^2
    X = trainData.to_numpy() 
    y = trainTarget.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = RandomForestRegressor(random_state=42, **rfParams)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #now we can make predictions of the training data
    predictData = testData.to_numpy()
    predictions = reg.predict(predictData)

    #keep a running list of any predicted UG or UV values are above the threshold
    #hard coded threshold
    #record both and pass them back to the main code where UG or UV is chosen depending on the user's prev input
    highUG = [i for i, pred in enumerate(predictions) if pred > 34]
    highUV = [i for i, pred in enumerate(predictions) if pred > 38] 

    return mse, r2, highUG, highUV


def randomTreeXGBoost(trainData, trainTarget, testData, **XGrfPara):
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

    #train
    X = trainData.to_numpy() 
    y = trainTarget.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgbModel = xgb.XGBRegressor(**parameters)
    xgbModel.fit(X_train, y_train)
    y_pred = xgbModel.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #predict
    predictData = testData.to_numpy()
    predictions = xgbModel.predict(predictData)

    highUG = [i for i, pred in enumerate(predictions) if pred > 34]
    highUV = [i for i, pred in enumerate(predictions) if pred > 38] 

    return mse, r2, highUG, highUV
