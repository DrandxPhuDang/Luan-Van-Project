import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR


class Regression_predict:

    def __init__(self, path_file_data, test_size, model_regression, prepro_or_none, grid_search_or_none):
        # tao bien dung chung
        super().__init__()

        # get data
        df = pd.read_csv(path_file_data)
        list_features = df.iloc[:0, 12:]
        features = [f'{e}' for e in list_features]
        X = df[features]
        y = df['Brix']

        # function preprocessing
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

        # Preprocessing or not Preprocessing
        if prepro_or_none == "Prepro":
            pca = PCA()
            pca.fit_transform(scale(X))
            X = preprocessing_data(X)
            # Train test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                    random_state=0)
            self.X_reduced_train = pca.fit_transform(scale(self.X_train))
            self.X_reduced_test = pca.transform(scale(self.X_test))
        if prepro_or_none == "None":
            # Train test split
            self.X_reduced_train, self.X_reduced_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    random_state=0)

        # Input model for regression
        try:
            if model_regression == "Neural":
                if prepro_or_none == "None":
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
            # Support vector regression
            if model_regression == "SVR":
                self.model = SVR(kernel='rbf', C=1, epsilon=0.2)

                self.model.fit(self.X_reduced_train, self.y_train)

            # Random Forest Regression
            if model_regression == "RF":
                # Using grid_search or not
                if grid_search_or_none == "None":
                    self.model = RandomForestRegressor(n_estimators=70,
                                                       random_state=42,
                                                       max_depth=5)
                    self.model.fit(self.X_reduced_train, self.y_train)
                if grid_search_or_none == "Grid_search":
                    rf = RandomForestRegressor()
                    param_rf = {
                        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'max_depth': [None, 5, 10, 15],
                    }
                    self.grid_rf = GridSearchCV(estimator=rf, param_grid=param_rf, cv=5)
                    self.grid_rf.fit(self.X_reduced_train, self.y_train)
                    self.model = RandomForestRegressor(n_estimators=self.grid_rf.best_params_["n_estimators"],
                                                       random_state=42,
                                                       max_depth=self.grid_rf.best_params_["max_depth"])
                    self.model.fit(self.X_reduced_train, self.y_train)
                    print("n_estimators: ", self.grid_rf.best_params_["n_estimators"])
                    print("max_depth: ", self.grid_rf.best_params_["max_depth"])

            # Gaussian Process Regression
            if model_regression == "GR":
                self.model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)
                self.model.fit(self.X_reduced_train, self.y_train)

            # Stacking regression (SVR, RF, GR) => RF (main model)
            if model_regression == "Stacking":
                # Using grid_search or not
                if grid_search_or_none == "None":
                    base_models = [
                        ('SVR', SVR(kernel='rbf', C=1, epsilon=0.2)),
                        ('RF', RandomForestRegressor(n_estimators=70,
                                                     random_state=42,
                                                     max_depth=5)),
                        ('GR', GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1))
                    ]
                    self.model = RandomForestRegressor(n_estimators=70,
                                                       random_state=42,
                                                       max_depth=5)
                    # self.model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)

                    # self.model = SVR(kernel='rbf', C=1, epsilon=0.2)

                    self.model = StackingRegressor(estimators=base_models, final_estimator=self.model)
                    self.model.fit(self.X_reduced_train, self.y_train)
                if grid_search_or_none == "Grid_search":
                    rf = RandomForestRegressor()
                    param_rf = {
                        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'max_depth': [None, 5, 10, 15],
                    }
                    self.grid_rf = GridSearchCV(estimator=rf, param_grid=param_rf, cv=5)
                    self.grid_rf.fit(self.X_reduced_train, self.y_train)
                    print("n_estimators: ", self.grid_rf.best_params_["n_estimators"])
                    print("max_depth: ", self.grid_rf.best_params_["max_depth"])

                    base_models = [
                        ('SVR', SVR(kernel='rbf', C=1, epsilon=0.2)),
                        ('RF', RandomForestRegressor(n_estimators=self.grid_rf.best_params_["n_estimators"],
                                                     random_state=42,
                                                     max_depth=self.grid_rf.best_params_["max_depth"])),
                        ('GR', GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1))
                    ]
                    self.model = RandomForestRegressor(n_estimators=self.grid_rf.best_params_["n_estimators"],
                                                       random_state=42,
                                                       max_depth=self.grid_rf.best_params_["max_depth"])
                    self.model = StackingRegressor(estimators=base_models, final_estimator=self.model)
                    self.model.fit(self.X_reduced_train, self.y_train)

            # Predicted Values
            y_pred_test = self.model.predict(self.X_reduced_test)
            y_pred_train = self.model.predict(self.X_reduced_train)

            # Print score
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

            # Load_spectrum
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

        # Export notice when model is not found
        except:
            print(f"Notice: This {model_regression} model is not available")
            print(f"Notice: Models available: SVR, RF, GR, Neural and Stacking")


# Running program
if __name__ == "__main__":
    Path_File_Data = fr'D:\Luan Van\Data\Final_Data\Final_Data.csv'

    # List model regression available: SVR, RF, GR, Neural and Stacking
    Regression_predict(path_file_data=Path_File_Data, test_size=0.3,
                       model_regression="Stacking", prepro_or_none="Prepro", grid_search_or_none="None")
