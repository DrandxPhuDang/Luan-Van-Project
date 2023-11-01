import pandas as pd
from sklearn.ensemble import StackingRegressor
from Helper.Drand_grid_search import *
from Helper.Drand_started_helper import *


class Regression_predict:

    def __init__(self, path_file_data, test_size, model_regression, prepro_data, find_best_parameter, kernel_pca,
                 start_col_X):
        self.prepro_data = prepro_data
        self.kernel_pca = kernel_pca
        super().__init__()

        # ------------------------------------------------- Import Data ------------------------------------------------
        df = pd.read_csv(path_file_data)
        X, y, features = get_data_X_y(df, start_col=start_col_X)

        # ----------------------------------------------- Preprocessing Data -------------------------------------------
        self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = train_test_split_kennard_stone(X, y, test_size, prepro_data)

        # ----------------------------------------------- Reduce Features Data -----------------------------------------
        self.X_train, self.X_val, self.X_test = reduce_kernel_pca(self.X_train, self.X_val, self.X_test, self.y_train,
                                                                  features_col=features, kernel_pca=kernel_pca)

        # ----------------------------------------------- Notice Model Input -------------------------------------------
        warning(model_regression=model_regression)

        # ---------------------------------------- KNeighbor Regression ------------------------------------------------
        '''K_Neighbor regression'''
        if model_regression == "KNN":
            self.name_model = 'K_Neighbor'
            if find_best_parameter is True:
                best_model_knn = Gridsearch_knn(self.X_val, self.y_val)
                print(best_model_knn.best_params_)
                print('Best score:', best_model_knn.best_score_)
                self.model = best_model_knn.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = KNeighborsRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- GB Regression -------------------------------------------------
        '''Gradient boosting regression'''
        if model_regression == "GBR":
            self.name_model = 'Gradient Boosting'
            if find_best_parameter is True:
                best_model_gbr = Gridsearch_gbr(self.X_val, self.y_val)
                print(best_model_gbr.best_params_)
                print('Best score:', best_model_gbr.best_score_)
                self.model = best_model_gbr.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = GradientBoostingRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- PLS Regression -------------------------------------------------
        '''PLS regression'''
        if model_regression == "PLS":
            self.name_model = 'PLS'
            if find_best_parameter is True:
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val, features=self.X_train.iloc[0, 0:])
                print(best_model_pls.best_params_)
                print('Best score:', best_model_pls.best_score_)
                self.model = best_model_pls.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = PLSRegression()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Ridge Regression -------------------------------------------------
        '''Ridge regression'''
        if model_regression == "R":
            self.name_model = 'Ridge'
            if find_best_parameter is True:
                best_model_r = Gridsearch_r(self.X_val, self.y_val)
                print(best_model_r.best_params_)
                print('Best score: ', best_model_r.best_score_)
                self.model = best_model_r.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = Ridge()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Lasso Regression -------------------------------------------------
        '''Lasso regression'''
        if model_regression == "L":
            self.name_model = 'Lasso'
            if find_best_parameter is True:
                best_model_l = Gridsearch_l(self.X_val, self.y_val)
                print(best_model_l.best_params_)
                print('Best score: ', best_model_l.best_score_)
                self.model = best_model_l.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = Ridge()
                self.model.fit(self.X_train, self.y_train)

        # ---------------------------------------------- XG_boost ------------------------------------------------------
        '''XGBoost regression'''
        if model_regression == "XGB":
            self.name_model = 'XGBoost'
            if find_best_parameter is True:
                best_model_xgb = Gridsearch_xgb(self.X_val, self.y_val)
                print(best_model_xgb.best_params_)
                print('Best score:', best_model_xgb.best_score_)
                self.model = best_model_xgb.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = xgb.XGBRegressor()
                self.model.fit(self.X_train, self.y_train)

        # -------------------------------------------- Neural ANN ------------------------------------------------------
        '''MLP regression'''
        if model_regression == "ANN":
            self.name_model = 'MLP'
            if find_best_parameter is True:
                best_model_ann = Gridsearch_ann(self.X_val, self.y_val)
                print(best_model_ann.best_params_)
                print('Best score:', best_model_ann.best_score_)
                self.model = best_model_ann.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            else:
                self.model = MLPRegressor()
                self.model.fit(self.X_train, self.y_train)

        # --------------------------------------------- Support Vector Machine -----------------------------------------
        '''Support vector regression'''
        if model_regression == "SVR":
            self.name_model = 'Support Vector'
            if find_best_parameter is True:
                best_model_svr = Gridsearch_svr(self.X_val, self.y_val)
                print(best_model_svr.best_params_)
                print('Best score:', best_model_svr.best_score_)
                self.model = best_model_svr.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            else:
                self.model = SVR()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Random Forest Regression -----------------------------------------
        '''Random Forest Regression'''
        if model_regression == "RF":
            self.name_model = 'Random Forest'
            if find_best_parameter is True:
                best_model_rf = Gridsearch_rf(self.X_val, self.y_val)
                print(best_model_rf.best_params_)
                print('Best score:', best_model_rf.best_score_)
                self.model = best_model_rf.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            else:
                self.model = RandomForestRegressor(random_state=42)
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Decision Tree Regression -----------------------------------------
        '''Decision Tree Regression'''
        if model_regression == "DT":
            self.name_model = 'Decision Tree'
            if find_best_parameter is True:
                best_model_dt = Gridsearch_dt(self.X_val, self.y_val)
                print(best_model_dt.best_params_)
                print('Best score:', best_model_dt.best_score_)
                self.model = best_model_dt.best_estimator_
                self.model.fit(self.X_train, self.y_train)

            else:
                self.model = DecisionTreeRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Linear Regression ------------------------------------------------
        '''Linear Regression'''
        if model_regression == "LR":
            self.name_model = 'Linear'
            if find_best_parameter is True:
                best_model_lr = Gridsearch_lr(self.X_val, self.y_val)
                print(best_model_lr.best_params_)
                print('Best score:', best_model_lr.best_score_)
                self.model = best_model_lr.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = LinearRegression()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Stacking Regression ----------------------------------------------
        '''Stacking regression'''
        if model_regression == "Stacking":
            self.name_model = 'Stacking'
            if find_best_parameter is True:
                # -------------------------------------- Grid Search ---------------------------------------------------
                best_model_rf = Gridsearch_rf(self.X_val, self.y_val)
                best_model_r = Gridsearch_r(self.X_val, self.y_val)
                best_model_xgb = Gridsearch_xgb(self.X_val, self.y_val)
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val, features=self.X_train.iloc[0, 0:])
                best_model_svr = Gridsearch_svr(self.X_val, self.y_val)

                # ----------------------------------- Print Best parameter ---------------------------------------------

                # -------------------------------------- Running model -------------------------------------------------
                base_models = [
                    ('svr', best_model_svr.best_estimator_),
                    ('rf', best_model_rf.best_estimator_),
                    ('gbr', best_model_xgb.best_estimator_),
                    ('pls', best_model_pls.best_estimator_)
                ]
                self.model = best_model_r.best_estimator_
                self.model = StackingRegressor(estimators=base_models, final_estimator=self.model,
                                               cv=KFold(n_splits=10, shuffle=True, random_state=42), verbose=10)
                self.model.fit(self.X_train, self.y_train)

            else:
                base_models = [
                    ('svr', SVR()),
                    ('rf', RandomForestRegressor()),
                    ('xgb', xgb.XGBRegressor()),
                    ('pls', PLSRegression())
                ]
                self.model = Ridge()
                self.model = StackingRegressor(estimators=base_models, final_estimator=self.model,
                                               cv=KFold(n_splits=10, shuffle=True, random_state=42), verbose=10)
                self.model.fit(self.X_train, self.y_train)

        # ---------------------------------------------- Predicted Values ----------------------------------------------
        '''Predicted Values'''
        y_pred_test = self.model.predict(self.X_test)
        y_pred_train = self.model.predict(self.X_train)

        # ------------------------------------------------ Print Score -------------------------------------------------
        print('--------------- TRAIN--------------------')
        print_score(self.y_train, y_pred_train)
        print('--------------- TEST--------------------')
        score_test = print_score(self.y_test, y_pred_test)
        print('--------------- RPD--------------------')
        RPD_Test = cal_rpd(self.y_test, y_pred_test)
        print('RPD:', "{:.2f}".format(RPD_Test))

        # ------------------------------------------------ Load Spectrum -----------------------------------------------
        load_spectrum(self.y_test, y_pred_test, name_model=self.name_model, score_test=score_test)
