#hyperparameters.py
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import numpy as np

#linear regression cannot be tuned any further.

def optimiseGB(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

    #split once: cross-validation will handle any training/validation sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    #set up hyperparameter grid
    paramGrid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }

    gbr = GradientBoostingRegressor(random_state=42)
    gridSearch = GridSearchCV(
        estimator=gbr, param_grid=paramGrid,
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gridSearch.fit(X_train, y_train)

    bestGBPara = gridSearch.best_estimator_
    bestGBScore = np.sqrt(-gridSearch.best_score_)

    return bestGBPara, bestGBScore


def optimiseRF(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

    #split once: cross-validation will handle any training/validation sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    #set up hyperparameter grid
    paramGrid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    #init model + run grid search
    rf = RandomForestRegressor(random_state=42)

    gridSearch = GridSearchCV(
        estimator=rf,
        param_grid=paramGrid,
        cv=3,                #3-fold cross validation
        n_jobs=-1,           #use all processors -> can be changed if have access to more processors
        scoring='neg_mean_squared_error',
        verbose=2
    )

    gridSearch.fit(X_train, y_train)

    bestRFPara = gridSearch.best_params_
    bestRFScore = np.sqrt(-gridSearch.best_score_)  #RMSE from neg MSE

    return bestRFPara, bestRFScore

def optimiseXGrf(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    paramGrid = {
        'max_depth': [3, 5, 6, 8],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'subsample': [0.3, 0.5, 0.7],
        'learning_rate': [0.5, 1.0],
    }

    baseModel = xgb.XGBRegressor(
        n_estimators=1,      # single tree
        random_state=42,
        booster='gbtree',
        objective='reg:squarederror'
    )

    gridSearch = GridSearchCV(
        estimator=baseModel,
        param_grid=paramGrid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )

    gridSearch.fit(X_train, y_train)

    bestXGrfPara =  gridSearch.best_params_
    bestXGrfModel =  gridSearch.best_estimator_

    y_pred = bestXGrfModel.predict(X_test)
    bestXGrfScore = root_mean_squared_error(y_test, y_pred)  #RMSE
    return bestXGrfPara, bestXGrfScore