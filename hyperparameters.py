#hyperparameters.py
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np



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