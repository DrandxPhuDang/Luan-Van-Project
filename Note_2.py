import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn import model_selection, preprocessing
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR


class Regression_predict:

    def __init__(self, Path_file_data, test_size, stacking_or_none):
        # tao bien dung chung
        super().__init__()

        df = pd.read_csv(Path_file_data)
        list_features = df.iloc[:0, 12:]
        features = [f'{e}' for e in list_features]
        X = df[features]
        y = df['Brix']

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

        X = preprocessing_data(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # scale predictor variables
        pca = PCA()
        pca.fit_transform(scale(X))
        X_reduced_train = pca.fit_transform(scale(X_train))
        X_reduced_test = pca.transform(scale(X_test))

        # rf = RandomForestRegressor()
        # param_rf = {
        #     'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        #     'max_depth': [None, 5, 10, 15],
        #     'min_samples_split': [2, 5, 10]
        # }
        # grid_rf = GridSearchCV(estimator=rf, param_grid=param_rf, cv=5)
        # grid_rf.fit(X_train, y_train)

        # print(grid_rf.best_params_["n_estimators"])
        # print(grid_rf.best_params_["max_depth"])
        # print(grid_rf.best_params_["min_samples_split"])

        if stacking_or_none == "None":
            # self.model = RandomForestRegressor(n_estimators=70,
            #                                    random_state=42,
            #                                    max_depth=5)

            # self.model = SVR(kernel='rbf', C=1, epsilon=0.2)

            poly = PolynomialFeatures(degree=2)
            X_reduced_train = poly.fit_transform(X_reduced_train)

            self.model = LinearRegression()

            self.model.fit(X_reduced_train, y_train)

        if stacking_or_none == "Stacking":
            base_models = [
                ('p', RandomForestRegressor(n_estimators=70,
                                            random_state=42,
                                            max_depth=5)),
                ('g', GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1))
            ]
            self.model = RandomForestRegressor(n_estimators=70,
                                               random_state=42,
                                               max_depth=5)
            # model = LinearRegression()
            self.model = StackingRegressor(estimators=base_models, final_estimator=self.model)
            self.model.fit(X_reduced_train, y_train)

        y_pred_test = self.model.predict(X_reduced_test)
        y_pred_train = self.model.predict(X_reduced_train)

        def print_score(y_actual, y_predicted):
            # R, R_Squared, R_MSE
            R_Train_pls = np.corrcoef(y_actual, y_predicted, rowvar=False)
            print('R:', "{:.3f}".format(R_Train_pls[0][1]))
            R_Squared_Train_pls = r2_score(y_actual, y_predicted)
            print('R^2:', "{:.3f}".format(R_Squared_Train_pls))
            print(f"Accuracy: {R_Squared_Train_pls * 100:.3f}%")
            RMSE_Train_pls = math.sqrt(mean_squared_error(y_actual, y_predicted))
            print('R_MSE :', "{:.3f}".format(RMSE_Train_pls))
            MAE_Train_pls = mean_absolute_error(y_actual, y_predicted)
            print('MAE:', "{:.3f}".format(MAE_Train_pls))
            return R_Train_pls, R_Squared_Train_pls, RMSE_Train_pls, MAE_Train_pls

        print('--------------- TRAIN--------------------')
        print_score(y_train, y_pred_train)
        print('--------------- TEST--------------------')
        score_test = print_score(y_test, y_pred_test)

        def load_spectrum(y_actual, y_pred):
            plt.scatter(y_actual, y_pred, label='Data')
            plt.xlabel('Actual Response')
            plt.ylabel('Predicted Response')
            plt.title(f'RF Regression (R²={score_test[1]:.2f})')
            reg_pls = np.polyfit(y_actual, y_pred, deg=1)
            trend_pls = np.polyval(reg_pls, y_actual)
            plt.plot(y_actual, trend_pls, 'r', label='Line pred')
            plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
            plt.show()

        load_spectrum(y_test, y_pred_test)


if __name__ == "__main__":
    path_file_data = fr'D:\Luan Van\Data\Final_Data\Final_Data.csv'
    Regression_predict(Path_file_data=path_file_data, test_size=0.3,
                       stacking_or_none="None")
