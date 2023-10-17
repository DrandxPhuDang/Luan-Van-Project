import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn import preprocessing
from kennard_stone import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import xgboost as xgb
from joblib import dump


class Regression_predict:

    def __init__(self, path_file_data, test_size, model_regression, prepro_data, find_best_parameter):
        # tao bien dung chung
        super().__init__()
        # ------------------------------------------------- Import Data ------------------------------------------------
        # get data
        df = pd.read_csv(path_file_data)
        list_features = df.iloc[:0, 12:]
        features = [f'{e}' for e in list_features]
        X = df[features]
        y = df['Brix']

        # ----------------------------------------------- Function defined ---------------------------------------------
        # Function Preprocessing Data
        def preprocessing_data(X_data):
            X_data = pd.DataFrame(X_data).dropna()
            # Fill in missing values using mean imputation
            X_data.fillna(X_data.mean(), inplace=True)
            # normalize
            X_data = preprocessing.normalize(X_data.values)
            # Normalizer
            prepro_normal_train = preprocessing.Normalizer().fit(X_data)
            X_data = prepro_normal_train.transform(X_data)
            # S_Filter
            X_data = savgol_filter(X_data, window_length=12, polyorder=1)
            # Min Max Scaler
            scaler_X_train = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            X_data = scaler_X_train.fit_transform(X_data)
            X_data = scaler_X_train.inverse_transform(X_data)
            X_data = pd.DataFrame(X_data)
            return X_data

        # ----------------------------------------------- Preprocessing Data -------------------------------------------
        # Preprocessing or not Preprocessing
        if prepro_data == "Prepro":
            # PCA for Dimensional reduction data
            pca = PCA()
            pca.fit_transform(scale(X))
            # Preprocessing data
            X = preprocessing_data(X)
            # Train test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
            # Fit on PCA
            self.X_reduced_train = pca.fit_transform(scale(self.X_train))
            self.X_reduced_test = pca.transform(scale(self.X_test))
        if prepro_data == "None":
            # Train test split
            self.X_reduced_train, self.X_reduced_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                                    test_size=test_size)

        # ----------------------------------------------- Notice Model Input -------------------------------------------
        list_model = ["SVR", "RF", "Stacking", "ANN", "R", "L", "XGB", "PLS"]
        cnt = 0
        for model in list_model:
            cnt += 1
            if model_regression == model:
                print(f"Notice: Model {model_regression} regression is available")
                print("Creating and training model...")
                break
            else:
                if cnt == len(list_model):
                    print(f"Notice: Model {model_regression} regression is not available")
                    print(f"Notice: Available model are: SVR, RF, GR, ANN, Stacking ")

        # ------------------------------------------- PLS Regression -------------------------------------------------
        if model_regression == "PLS":
            self.model = PLSRegression(n_components=120)
            self.model.fit(self.X_reduced_train, self.y_train)

        # ------------------------------------------- Ridge Regression -------------------------------------------------
        if model_regression == "R":
            self.model = Ridge(alpha=24)
            self.model.fit(self.X_reduced_train, self.y_train)

        # ------------------------------------------- Lasso Regression -------------------------------------------------
        if model_regression == "L":
            self.model = Lasso(alpha=0.01)
            self.model.fit(self.X_reduced_train, self.y_train)

        # ---------------------------------------------- XG_boost ------------------------------------------------------
        if model_regression == "XGB":
            self.model = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.01,
                                          subsample=0.7, colsample_bytree=1)

            self.model.fit(self.X_reduced_train, self.y_train)

        # -------------------------------------------- Neural ANN ------------------------------------------------------
        if model_regression == "ANN":
            if prepro_data == "None":
                scaler = StandardScaler()
                self.X_reduced_train = scaler.fit_transform(self.X_reduced_train)
                self.X_reduced_test = scaler.transform(self.X_reduced_test)
            self.model = MLPRegressor(
                activation='logistic',
                hidden_layer_sizes=(800, 175),
                alpha=0.001,
                random_state=20,
                early_stopping=False,
                solver='adam'
            )
            self.model.fit(self.X_reduced_train, self.y_train)

        # --------------------------------------------- Support Vector Machine -----------------------------------------
        # Support vector regression
        if model_regression == "SVR":
            self.model = SVR(kernel='rbf', C=2, epsilon=0.01)
            self.model.fit(self.X_reduced_train, self.y_train)

        # ------------------------------------------- Random Forest Regression -----------------------------------------
        # Random Forest Regression
        if model_regression == "RF":
            self.model = RandomForestRegressor(n_estimators=70, random_state=42, max_depth=5)
            self.model.fit(self.X_reduced_train, self.y_train)

        # ------------------------------------------- Stacking Regression ----------------------------------------------
        # Stacking regression (SVR, RF, GR) => RF (main model)
        if model_regression == "Stacking":
            # Using grid_search or not
            if find_best_parameter == "None":
                pass
            if find_best_parameter == "Find":
                pass

        # ---------------------------------------------- Predicted Values ----------------------------------------------
        print(f"Successful Creating a {model_regression} model")

        # Predicted Values
        y_pred_test = self.model.predict(self.X_reduced_test)
        y_pred_train = self.model.predict(self.X_reduced_train)
        print("Evaluating Regression Models")

        custom_path = r"D:\Luan Van/model_predicted.pkl"
        with open(custom_path, 'wb') as file:
            pickle.dump(self.model, file)

        # ------------------------------------------------ Print Score -------------------------------------------------
        def print_score(y_actual, y_predicted):
            # R, R_Squared, R_MSE
            R = np.corrcoef(y_actual, y_predicted, rowvar=False)
            print('R:', "{:.3f}".format(R[0][1]))
            R_Squared = r2_score(y_actual, y_predicted)
            print('R^2:', "{:.3f}".format(R_Squared))
            print(f"Accuracy: {R_Squared * 100:.3f}%")
            R_MSE = math.sqrt(mean_squared_error(y_actual, y_predicted))
            print('R_MSE :', "{:.3f}".format(R_MSE))
            MAE = mean_absolute_error(y_actual, y_predicted)
            print('MAE:', "{:.3f}".format(MAE))
            return R, R_Squared, R_MSE, MAE

        print('--------------- TRAIN--------------------')
        print_score(self.y_train, y_pred_train)
        print('--------------- TEST--------------------')
        score_test = print_score(self.y_test, y_pred_test)
        print('--------------- RPD--------------------')
        RPD_Test = np.std(self.y_test) / score_test[2]
        print('RPD:', "{:.2f}".format(RPD_Test))

        # ------------------------------------------------ Load Spectrum -----------------------------------------------
        def load_spectrum(y_actual, y_pred):
            plt.scatter(y_actual, y_pred, label='Data')
            plt.xlabel('Actual Response')
            plt.ylabel('Predicted Response')
            plt.title(f'Regression (R²={score_test[1]:.2f})')
            reg_pls = np.polyfit(y_actual, y_pred, deg=1)
            trend_pls = np.polyval(reg_pls, y_actual)
            plt.plot(y_actual, trend_pls, 'r', label='Line pred')
            plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
            plt.show()

        load_spectrum(self.y_test, y_pred_test)
