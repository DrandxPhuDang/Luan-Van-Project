import math
from sys import stdout
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kennard_stone import train_test_split
from scipy.signal import savgol_filter
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

data_pre_train = pd.read_csv(fr'D:\Luan Van\Data\train_test' + r'\prepro_train' + '.csv')
data_pre_test = pd.read_csv(fr'D:\Luan Van\Data\train_test' + r'\prepro_test' + '.csv')

listWavelength = data_pre_train.iloc[:0, 1:]
features = [f'{e}' for e in listWavelength]

X_pre_train = data_pre_train.iloc[:, 1:]
y_pre_train = data_pre_train['Brix']
X_pre_test = data_pre_test.iloc[:, 1:]
y_pre_test = data_pre_test['Brix']

# choose number component fit by Cross-validation
param = {
    'n_components': [3]}
model_pls = PLSRegression()
search = GridSearchCV(model_pls, param, cv=10, scoring='neg_mean_squared_error', return_train_score=True,
                      refit=True)
search.fit(X_pre_train, y_pre_train)

model_rf = RandomForestRegressor()
# Define the hyperparameters for grid search
param_grid = {
    'n_estimators': [59]
}
# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=10)
grid_search.fit(X_pre_train, y_pre_train)

param_grid_knn = {'n_neighbors': [20]}
knn = KNeighborsRegressor()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=10)
grid_knn.fit(X_pre_train, y_pre_train)

print(grid_search.best_params_["n_estimators"])
print(search.best_params_["n_components"])
print(grid_knn.best_params_["n_neighbors"])

base_models = [
    ('rf', RandomForestRegressor(n_estimators=grid_search.best_params_["n_estimators"],
                                 random_state=42)),
    ('pls', PLSRegression(n_components=search.best_params_["n_components"]))
]

model_knn = KNeighborsRegressor(n_neighbors=grid_knn.best_params_["n_neighbors"])

model_knn = StackingRegressor(estimators=base_models, final_estimator=model_knn)

model_knn.fit(X_pre_train, y_pre_train)

y_train_pred_knn = model_knn.predict(X_pre_train)
y_test_pred_knn = model_knn.predict(X_pre_test)

# R, R_Squared, R_MSE
print('--------------- TRAIN--------------------')
R_Train_knn = np.corrcoef(y_pre_train, y_train_pred_knn, rowvar=False)
print('R:', "{:.3f}".format(R_Train_knn[0][1]))
R_Squared_Train_knn = r2_score(y_pre_train, y_train_pred_knn)
print('R^2:', "{:.3f}".format(R_Squared_Train_knn))
print(f"Accuracy: {R_Squared_Train_knn * 100:.3f}%")
RMSE_Train_knn = math.sqrt(mean_squared_error(y_pre_train, y_train_pred_knn))
print('R_MSE :', "{:.3f}".format(RMSE_Train_knn))
MAE_Train_knn = mean_absolute_error(y_pre_train, y_train_pred_knn)
print('MAE:', "{:.3f}".format(MAE_Train_knn))
# R, R_Squared, R_MSE
print('--------------- TEST--------------------')
R_Test_knn = np.corrcoef(y_pre_test, y_test_pred_knn, rowvar=False)
print('R:', "{:.3f}".format(R_Test_knn[0][1]))
R_Squared_Test_knn = r2_score(y_pre_test, y_test_pred_knn)
print('R^2:', "{:.3f}".format(R_Squared_Test_knn))
print(f"Accuracy: {R_Squared_Test_knn * 100:.3f}%")
RMSE_Test_knn = math.sqrt(mean_squared_error(y_pre_test, y_test_pred_knn))
print('R_MSE :', "{:.3f}".format(RMSE_Test_knn))
MAE_Test_knn = mean_absolute_error(y_pre_test, y_test_pred_knn)
print('MAE:', "{:.3f}".format(MAE_Test_knn))
print('--------------- RPD--------------------')
RPD_Test_knn = np.std(y_pre_test) / RMSE_Test_knn
print('RPD:', "{:.2f}".format(RPD_Test_knn))


def load_spectrum():
    plt.scatter(y_pre_test, y_test_pred_knn)
    plt.xlabel('Actual Response')
    plt.ylabel('Predicted Response')
    plt.title(f' Regression (R²={R_Squared_Test_knn:.2f})')
    reg_knn = np.polyfit(y_pre_test, y_test_pred_knn, deg=1)
    trend_knn = np.polyval(reg_knn, y_pre_test)
    plt.plot(y_pre_test, trend_knn, 'r')
    plt.plot(y_pre_test, y_pre_test, color='green', linestyle='--', linewidth=1)
    plt.show()


load_spectrum()
