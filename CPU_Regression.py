import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

df = pd.read_excel('./Dataset/dataset.xlsx')

X = df.iloc[:, 1:12].values
y = df.iloc[:,-1].values
y = y.reshape(-1, 1) 

#encoding
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

best_ratio = 0.3
scaler = 1

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=best_ratio)
X_train = pd.read_excel('./Dataset/Distributed/X_train.xlsx').values
X_test = pd.read_excel('./Dataset/Distributed/X_test.xlsx').values
y_train = pd.read_excel('./Dataset/Distributed/y_train.xlsx').values
y_test = pd.read_excel('./Dataset/Distributed/y_test.xlsx').values


#feature scaling - 1 for minmax 0 for S.S
if(scaler == 1):
    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
else:
    sc_X = StandardScaler()
    sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.transform(y_test)


#applying Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#applying Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X_train, y_train.ravel())
y_pred_rf = rf_reg.predict(X_test)

#applying Neural Network 
"""
# parameter grid for tuning
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (100,)],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.01],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.0001, 0.001],
}

nn_reg = MLPRegressor(max_iter=78, early_stopping=True, validation_fraction=0.1, random_state=42)

# grid search with cross-validation
grid_search = GridSearchCV(nn_reg, param_grid, cv=5)
grid_search.fit(X_train, y_train.ravel())

# print best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

nn_reg = MLPRegressor(hidden_layer_sizes=(10), max_iter=78, solver='lbfgs',verbose=False, learning_rate='constant', learning_rate_init=0.01,random_state=42)
nn_reg.fit(X_train, y_train.ravel())
y_pred_nn = nn_reg.predict(X_test)

#applying Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=143, learning_rate=0.1,random_state=42, max_depth=1)
gbr.fit(X_train, y_train.ravel())
y_pred_gbr = gbr.predict(X_test)

#applying Support Vector Regression
"""
#HyperParameter Tuning for SVR
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'sigmoid']
}

grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train.ravel())

best_svr = grid.best_estimator_
y_pred_svr = best_svr.predict(X_test)

# Print the best hyperparameter
print("Best parameters:", grid.best_params_)"""

svr = SVR(C=1000, gamma=1,kernel='linear')
svr.fit(X_train, y_train.ravel())
y_pred_svr = svr.predict(X_test)

# Metrics for Linear Regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print('\nMultiple Linear Regression')
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R^2:', r2)

# Metrics for Random Forest Regression
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print('\nRandom Forest Regression')
print('MAE:', mae_rf)
print('MSE:', mse_rf)
print('RMSE:', rmse_rf)
print('R^2:', r2_rf)

# Metrics for Neural Network Regression
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = mse ** 0.5
r2_nn = r2_score(y_test, y_pred_nn)

print('\nNeural Network Regression:')
print('MAE:', mae_nn)
print('MSE:', mse_nn)
print('RMSE:', rmse_nn)
print('R^2:', r2_nn)

# Metrics for Gradient Boosting Regression
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = mse_gbr ** 0.5
r2_gbr = r2_score(y_test, y_pred_gbr)

print('\nGradient Boosting Regression')
print('MAE:', mae_gbr)
print('MSE:', mse_gbr)
print('RMSE:', rmse_gbr)
print('R^2:', r2_gbr)

# Metrics for Support Vector Regression
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = mse_svr ** 0.5
r2_svr = r2_score(y_test, y_pred_svr)

print('\nSupport Vector Regression')
print('MAE:', mae_svr)
print('MSE:', mse_svr)
print('RMSE:', rmse_svr)
print('R^2:', r2_svr)
