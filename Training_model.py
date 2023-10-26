import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from kennard_stone import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from GridSearch_Model import Gridsearch_ann, Gridsearch_rf, Gridsearch_svr, Gridsearch_r, Gridsearch_knn, \
    Gridsearch_dt, Gridsearch_pls, Gridsearch_xgb, Gridsearch_gbr, Gridsearch_lr, Gridsearch_l


class Regression_predict:

    def __init__(self, path_file_data, test_size, model_regression, prepro_data, find_best_parameter):
        # tao bien dung chung
        super().__init__()
        # ------------------------------------------------- Import Data ------------------------------------------------
        '''get data'''
        df = pd.read_csv(path_file_data)
        list_features = df.iloc[:0, 12:]
        features = [f'{e}' for e in list_features]
        X = df[features]
        y = df['Brix']

        # ----------------------------------------------- Function defined ---------------------------------------------
        '''Function Preprocessing Data'''

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
        '''Preprocessing or not Preprocessing'''
        if prepro_data == "Prepro":
            X = preprocessing_data(X)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.2)
            scaler_nor = StandardScaler()
            self.X_train = scaler_nor.fit_transform(self.X_train)
            self.X_val = scaler_nor.transform(self.X_val)
            self.X_test = scaler_nor.transform(self.X_test)
        if prepro_data == "None":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.2)

        # ----------------------------------------------- Notice Model Input -------------------------------------------
        list_model = ["SVR", "RF", "Stacking", "ANN", "R", "L", "XGB", "PLS", "GBR", "KNN", "DT", "LR"]
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
                    print(f"Available model: SVR, RF, PLS, ANN, R, XGB, PLS, DT, LR, KNN, GBR and Stacking")

        # ---------------------------------------- KNeighbor Regression ------------------------------------------------
        '''K_Neighbor regression'''
        if model_regression == "KNN":
            name_model = 'K_Neighbor'
            if find_best_parameter == "Find":
                best_model_knn = Gridsearch_knn(self.X_val, self.y_val)
                print('n_neighbors:', best_model_knn.best_params_["n_neighbors"])
                print('leaf_size:', best_model_knn.best_params_["leaf_size"])
                print('Best score:', best_model_knn.best_score_)
                self.model = best_model_knn.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = KNeighborsRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- GB Regression -------------------------------------------------
        '''Gradient boosting regression'''
        if model_regression == "GBR":
            name_model = 'Gradient Boosting'
            if find_best_parameter == "Find":
                best_model_gbr = Gridsearch_gbr(self.X_val, self.y_val)
                print('n_estimators:', best_model_gbr.best_params_["n_estimators"])
                print('learning_rate:', best_model_gbr.best_params_["learning_rate"])
                print('max_depth:', best_model_gbr.best_params_["max_depth"])
                print('Best score:', best_model_gbr.best_score_)
                self.model = best_model_gbr.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = GradientBoostingRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- PLS Regression -------------------------------------------------
        '''PLS regression'''
        if model_regression == "PLS":
            name_model = 'PLS'
            list_len_features = []
            for value in range(1, len(features)):
                list_len_features.append(value)
            if find_best_parameter == "Find":
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val, features=features)
                print('n_components:', best_model_pls.best_params_["n_components"])
                print('Best score:', best_model_pls.best_score_)
                self.model = best_model_pls.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = PLSRegression()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Ridge Regression -------------------------------------------------
        '''Ridge regression'''
        if model_regression == "R":
            name_model = 'Ridge'
            if find_best_parameter == "Find":
                best_model_r = Gridsearch_r(self.X_val, self.y_val)
                print('alpha:', best_model_r.best_params_["alpha"])
                print('Best score: ', best_model_r.best_score_)
                self.model = best_model_r.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = Ridge()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Lasso Regression -------------------------------------------------
        '''Lasso regression'''
        if model_regression == "L":
            name_model = 'Lasso'
            if find_best_parameter == "Find":
                best_model_l = Gridsearch_l(self.X_val, self.y_val)
                print('alpha:', best_model_l.best_params_["alpha"])
                print('Best score: ', best_model_l.best_score_)
                self.model = best_model_l.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = Ridge()
                self.model.fit(self.X_train, self.y_train)

        # ---------------------------------------------- XG_boost ------------------------------------------------------
        '''XGBoost regression'''
        if model_regression == "XGB":
            name_model = 'XGBoost'
            if find_best_parameter == "Find":
                best_model_xgb = Gridsearch_xgb(self.X_val, self.y_val)
                print('learning_rate:', best_model_xgb.best_params_["learning_rate"])
                print('max_depth:', best_model_xgb.best_params_["max_depth"])
                print('colsample_bytree:', best_model_xgb.best_params_["colsample_bytree"])
                print('n_estimators:', best_model_xgb.best_params_["n_estimators"])
                print('Best score:', best_model_xgb.best_score_)
                self.model = best_model_xgb.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = xgb.XGBRegressor()
                self.model.fit(self.X_train, self.y_train)

        # -------------------------------------------- Neural ANN ------------------------------------------------------
        '''MLP regression'''
        if model_regression == "ANN":
            name_model = 'MLP'
            if find_best_parameter == "Find":
                best_model_ann = Gridsearch_ann(self.X_val, self.y_val)
                print('hidden_layer_sizes:', best_model_ann.best_params_["hidden_layer_sizes"])
                print('alpha:', best_model_ann.best_params_["alpha"])
                print('Best score:', best_model_ann.best_score_)
                self.model = best_model_ann.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            if find_best_parameter == "None":
                self.model = MLPRegressor()
                self.model.fit(self.X_train, self.y_train)

        # --------------------------------------------- Support Vector Machine -----------------------------------------
        '''Support vector regression'''
        if model_regression == "SVR":
            name_model = 'Support Vector'
            if find_best_parameter == "Find":
                best_model_svr = Gridsearch_svr(self.X_val, self.y_val)
                print('C:', best_model_svr.best_params_["C"])
                print('epsilon:', best_model_svr.best_params_["epsilon"])
                print('degree:', best_model_svr.best_params_["degree"])
                print('cache_size:', best_model_svr.best_params_["cache_size"])
                print('Best score:', best_model_svr.best_score_)
                self.model = best_model_svr.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            if find_best_parameter == "None":
                self.model = SVR()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Random Forest Regression -----------------------------------------
        '''Random Forest Regression'''
        if model_regression == "RF":
            name_model = 'Random Forest'
            if find_best_parameter == "Find":
                best_model_rf = Gridsearch_rf(self.X_val, self.y_val)
                print('n_estimators:', best_model_rf.best_params_["n_estimators"])
                print('max_depth:', best_model_rf.best_params_["max_depth"])
                print('Best score:', best_model_rf.best_score_)
                self.model = best_model_rf.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            if find_best_parameter == "None":
                self.model = RandomForestRegressor(random_state=42)
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Decision Tree Regression -----------------------------------------
        '''Decision Tree Regression'''
        if model_regression == "DT":
            name_model = 'Decision Tree'
            if find_best_parameter == "Find":
                best_model_dt = Gridsearch_dt(self.X_val, self.y_val)
                print('criterion:', best_model_dt.best_params_["criterion"])
                print('max_depth:', best_model_dt.best_params_["max_depth"])
                print('min_samples_split:', best_model_dt.best_params_["min_samples_split"])
                print('max_leaf_nodes:', best_model_dt.best_params_["max_leaf_nodes"])
                print('Best score:', best_model_dt.best_score_)
                self.model = best_model_dt.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            if find_best_parameter == "None":
                self.model = DecisionTreeRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Linear Regression ------------------------------------------------
        '''Linear Regression'''
        if model_regression == "LR":
            name_model = 'Linear'
            if find_best_parameter == "Find":
                best_model_lr = Gridsearch_lr(self.X_val, self.y_val)
                print('fit_intercept:', best_model_lr.best_params_["fit_intercept"])
                print('Best score:', best_model_lr.best_score_)
                self.model = best_model_lr.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            if find_best_parameter == "None":
                self.model = LinearRegression()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Stacking Regression ----------------------------------------------
        '''Stacking regression'''
        if model_regression == "Stacking":
            name_model = 'Stacking'
            if find_best_parameter == "Find":
                # -------------------------------------- Grid Search ---------------------------------------------------
                best_model_xgb = Gridsearch_xgb(self.X_train, self.y_train)
                best_model_rf = Gridsearch_rf(self.X_val, self.y_val)
                best_model_r = Gridsearch_r(self.X_val, self.y_val)
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val, features=features)

                # ----------------------------------- Print Best parameter ---------------------------------------------
                print('--------------------------XGBoosting-----------------------------')
                print('n_estimators:', best_model_xgb.best_params_["n_estimators"])
                print('max_depth:', best_model_xgb.best_params_["max_depth"])
                print('learning_rate:', best_model_xgb.best_params_["learning_rate"])
                print('colsample_bytree:', best_model_xgb.best_params_["colsample_bytree"])
                print('--------------------------Random Forest-----------------------------')
                print('n_estimators:', best_model_rf.best_params_["n_estimators"])
                print('max_depth:', best_model_rf.best_params_["max_depth"])
                print('-------------------------Ridge-----------------------------')
                print('alpha:', best_model_r.best_params_["alpha"])
                print('-------------------------PLS------------------------------')
                print('n_components:', best_model_pls.best_params_["n_components"])

                # -------------------------------------- Running model -------------------------------------------------
                base_models = [
                    ('xbg', best_model_xgb.best_estimator_),
                    ('rf', best_model_rf.best_estimator_),
                    ('pls', best_model_pls.best_estimator_),

                ]
                self.model = best_model_r.best_estimator_
                self.model = StackingRegressor(estimators=base_models, final_estimator=self.model,
                                               cv=len(base_models))
                self.model.fit(self.X_train, self.y_train)

            if find_best_parameter == "None":
                base_models = [
                    ('pls', PLSRegression(n_components=len(features))),
                    ('rf', RandomForestRegressor()),
                ]

                self.model = PLSRegression()
                self.model = StackingRegressor(estimators=base_models, final_estimator=self.model,
                                               cv=len(base_models))
                self.model.fit(self.X_train, self.y_train)

        print(f"Successful Creating a {model_regression} model")

        # ---------------------------------------------- Predicted Values ----------------------------------------------
        '''Predicted Values'''
        y_pred_test = self.model.predict(self.X_test)
        y_pred_train = self.model.predict(self.X_train)

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
            plt.title(f'{name_model} Regression (R²={score_test[1]:.2f})')
            reg_pls = np.polyfit(y_actual, y_pred, deg=1)
            trend_pls = np.polyval(reg_pls, y_actual)
            plt.plot(y_actual, trend_pls, 'r', label='Line pred')
            plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
            plt.show()

        load_spectrum(self.y_test, y_pred_test)
