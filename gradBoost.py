#gradBoost.py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def doGradBoost(trainData, trainTarget, testData,**gbParams):
    #train

    #features are stored in X
    #targets stored in y
    #changes the pandas df into numpy array
    X = trainData.to_numpy() 
    y = trainTarget.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gbr = GradientBoostingRegressor(**gbParams, random_state=42)
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #predict
    predictData = testData.to_numpy()
    predictions = gbr.predict(predictData)

    highUG = [i for i, pred in enumerate(predictions) if pred > 34]
    highUV = [i for i, pred in enumerate(predictions) if pred > 38] 

    return mse, r2, highUG, highUV
