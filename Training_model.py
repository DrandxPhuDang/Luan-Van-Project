from sklearn.ensemble import StackingRegressor
from Helper.Drand_grid_search import *
from Helper.Drand_preprocessing import *
from Helper.Drand_started_helper import *


class Regression_predict:

    def __init__(self, path_file_data, test_size, model_regression, prepro_data, find_best_parameter, kernel_pca,
                 start_col_X):
        super().__init__()

        # -------------------------------------------- Import Data -----------------------------------------------------
        """ Import Data """
        df = pd.read_csv(path_file_data)
        X, y, features = get_data_X_y(df, start_col=start_col_X)
        """ Preprocessing Data """
        self.X_train, self.X_val, self.X_test, \
            self.y_train, self.y_val, self.y_test = train_test_split_kennard_stone(X, y, test_size, prepro_data)
        """ Reduce Features Data """
        self.X_train, self.X_val, self.X_test = reduce_kernel_pca(self.X_train, self.X_val, self.X_test, self.y_train,
                                                                  features_col=features, kernel_pca=kernel_pca)

        # self.X_train, self.y_train = remove_outliers_model(self.X_train, self.y_train, threshold=2)
        # self.X_val, self.y_val = remove_outliers_model(self.X_val, self.y_val, threshold=2)
        # self.X_test, self.y_test = remove_outliers_model(self.X_test, self.y_test, threshold=2)

        """ warming Model Input """
        warning(model_regression=model_regression)
        # ---------------------------------------- ExtraTrees Regressor ------------------------------------------------
        '''ExtraTrees Regression'''
        if model_regression == "ETR":
            self.name_model = 'ExtraTrees'
            '''Find best parameter or None'''
            if find_best_parameter is True:
                best_model_etr = Gridsearch_etr(self.X_val, self.y_val)
                print(best_model_etr.best_params_)
                print('Best score:', best_model_etr.best_score_)
                self.model = best_model_etr.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = ExtraTreesRegressor()
                self.model.fit(self.X_train, self.y_train)

        # ---------------------------------------- KNeighbor Regression ------------------------------------------------
        '''K_Neighbor Regression'''
        if model_regression == "KNN":
            self.name_model = 'K_Neighbor'
            '''Find best parameter or None'''
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
        '''Gradient Boosting Regression'''
        if model_regression == "GBR":
            self.name_model = 'Gradient Boosting'
            '''Find best parameter or None'''
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
        '''PLS Regression'''
        if model_regression == "PLS":
            self.name_model = 'PLS'
            '''Find best parameter or None'''
            if find_best_parameter is True:
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val,
                                                features=pd.DataFrame(self.X_train).iloc[0, 0:])
                print(best_model_pls.best_params_)
                print('Best score:', best_model_pls.best_score_)
                self.model = best_model_pls.best_estimator_
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = PLSRegression()
                self.model.fit(self.X_train, self.y_train)

        # ------------------------------------------- Ridge Regression -------------------------------------------------
        '''Ridge Regression'''
        if model_regression == "R":
            self.name_model = 'Ridge'
            '''Find best parameter or None'''
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
        '''Lasso Regression'''
        if model_regression == "L":
            self.name_model = 'Lasso'
            '''Find best parameter or None'''
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
        '''XGBoost Regression'''
        if model_regression == "XGB":
            self.name_model = 'XGBoost'
            '''Find best parameter or None'''
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
        '''MLP Regression'''
        if model_regression == "ANN":
            self.name_model = 'MLP'
            '''Checking again with data preprocessing because MLPRegression request standardized data'''
            if prepro_data is None:
                Scaler = StandardScaler()
                self.X_train = Scaler.fit_transform(self.X_train)
                self.X_val = Scaler.transform(self.X_val)
                self.X_test = Scaler.transform(self.X_test)
            else:
                pass

            '''Find best parameter or None'''
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
        '''Support Vector Regression'''
        if model_regression == "SVR":
            self.name_model = 'Support Vector'
            '''Find best parameter or None'''
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
            '''Find best parameter or None'''
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
            '''Find best parameter or None'''
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
            '''Find best parameter or None'''
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
        '''Stacking Regression'''
        if model_regression == "Stacking":
            self.name_model = 'Stacking'
            '''Find best parameter or None'''
            if find_best_parameter is True:
                '''Grid Search'''
                best_model_etr = Gridsearch_etr(self.X_val, self.y_val)
                best_model_r = Gridsearch_r(self.X_val, self.y_val)
                best_model_xgb = Gridsearch_xgb(self.X_val, self.y_val)
                best_model_pls = Gridsearch_pls(self.X_val, self.y_val, features=pd.DataFrame(self.X_train).iloc[0, 0:])
                best_model_svr = Gridsearch_svr(self.X_val, self.y_val)
                '''Print Best parameter'''
                print_best_params([best_model_etr, best_model_r, best_model_xgb,
                                   best_model_pls, best_model_svr])
                '''Fitting model'''
                base_models = [
                    ('svr', best_model_svr.best_estimator_),
                    ('etr', best_model_etr.best_estimator_),
                    ('xgb', best_model_xgb.best_estimator_),
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
        '''Accuracy score'''
        print('--------------- TRAIN--------------------')
        print_score(self.y_train, y_pred_train)
        print('--------------- TEST--------------------')
        score_test = print_score(self.y_test, y_pred_test)
        print('--------------- RPD--------------------')
        RPD_Test = cal_rpd(self.y_test, y_pred_test)
        print('RPD:', "{:.2f}".format(RPD_Test))

        # ------------------------------------------------ Load Spectrum -----------------------------------------------
        '''Plot Regression'''
        load_spectrum(self.y_test, y_pred_test, name_model=self.name_model, score_test=score_test)
