#random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def doRandomForest(data, true, **rfParams):
    X = data.to_numpy() 
    y = true.iloc[:, 0].to_numpy().ravel()
    #changes the pandas df into numpy array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = RandomForestRegressor(random_state=42, **rfParams)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def randomTreeXGBoost(data,true, XGrfPara):
    X = data.to_numpy() 
    y = true.iloc[:, 0].to_numpy().ravel()

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=1,
        learning_rate=XGrfPara.get('learning_rate', 1.0),
        max_depth=XGrfPara.get('max_depth', 6),
        colsample_bytree=XGrfPara.get('colsample_bytree', 0.5),
        subsample=XGrfPara.get('subsample', 0.5),
        random_state=42,
        booster='gbtree',
        objective='reg:squarederror'
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
