import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

X = pd.read_csv('..\Kaggle Comptetion\Housing Prices/x.csv')
y = X.pop('SalePrice')

categorical_variables = list(X.dtypes[X.dtypes == 'object'].index)
numeric = list(X.dtypes[X.dtypes != 'object'].index)

for variable in categorical_variables:
    X[variable].fillna('Missing', inplace = True)
    dummies = pd.get_dummies(X[variable], prefix = variable)
    X = pd.concat([X, dummies], axis = 1)
    X.drop([variable], axis = 1, inplace = True)

for var in numeric:
    X[var].fillna(value = X[var].mean(), inplace = True)
#maybe using machine learning algorthm to predict na values rather than fill in with mea

model = RandomForestRegressor(100, oob_score=True, random_state = 42)
model.fit(X,y)

model.oob_score_ #0.8671249709442207

y_oob = model.oob_prediction_
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print mean_absolute_error(y, y_oob )
print mean_squared_error(y, y_oob )

param_grid = { 
    'n_estimators': [200, 500, 750, 1000],
    'max_features': ['auto', 'sqrt', 'log2', 'None'],
    'min_samples_leaf': [1,2,3]
}

CV_rfr = GridSearchCV(estimator=model, param_grid=param_grid, cv= 10)
CV_rfr.fit(X, y)
print CV_rfr.best_params_


import matplotlib.pyplot as plt
% matplotlib inline
feature_importances = pd.Series(model.feature_importances_, index = X.columns)
feature_importances.sort()
feature_importances.plot(kind = 'barh', figsize = (7,136)