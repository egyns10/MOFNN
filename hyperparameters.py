#hyperparameters.py
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np
import warnings
warnings.filterwarnings("ignore")  #suppresses all warnings generated within this file's code

#linear regression cannot be tuned any further.

def optimiseGB(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

    #set up hyperparameter grid
    paramGrid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }

    gbr = GradientBoostingRegressor(random_state=42)
    gridSearch = GridSearchCV(
        estimator=gbr,
        param_grid=paramGrid,
        cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose = 1
        )
    gridSearch.fit(X, y)

    bestGBPara = gridSearch.best_params_
    bestGBScore = np.sqrt(-gridSearch.best_score_)

    return bestGBPara, bestGBScore


def optimiseRF(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

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
        verbose=1
    )

    gridSearch.fit(X, y)

    bestRFPara = gridSearch.best_params_
    bestRFScore = np.sqrt(-gridSearch.best_score_)  #RMSE from neg MSE

    return bestRFPara, bestRFScore

def optimiseXGrf(data, true):
    X = data.to_numpy()
    y = true.iloc[:, 0].to_numpy().ravel()

    paramGrid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 6, 8],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'subsample': [0.3, 0.5, 0.7],
        'learning_rate': [0.5, 1.0],
    }

    xgrf = xgb.XGBRegressor(
        random_state=42,
        booster='gbtree',
        objective='reg:squarederror'
    )

    gridSearch = GridSearchCV(
        estimator=xgrf,
        param_grid=paramGrid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose = 1
    )

    gridSearch.fit(X, y)

    bestXGrfPara =  gridSearch.best_params_
    bestXGrfModel =  gridSearch.best_estimator_

    y_pred = bestXGrfModel.predict(X)
    bestXGrfScore = np.sqrt(-gridSearch.best_score_)
    return bestXGrfPara, bestXGrfScore