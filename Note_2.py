
# Chọn số component phù hợp bằng Cross-validation
param = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                     25]}
# param = {'n_components':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
plsr = PLSRegression()
search = GridSearchCV(plsr, param, cv=10, scoring='neg_mean_squared_error', return_train_score=True,
                      refit=True)
search.fit(X_Train_centered_pls, self.y_pre_train)
# Train model với thông số tìm được bằng cross-validation
plsr = PLSRegression(n_components=search.best_params_["n_components"])
plsr.fit(X_Train_centered_pls, self.y_pre_train)
# Evaluate model
y_train_pred_pls = plsr.predict(X_Train_centered_pls)
y_test_pred_pls = plsr.predict(X_Test_centered_pls)
# R, R_Squared, RMSE
print('--------------- TRAIN--------------------')
R_Train_pls = np.corrcoef(self.y_pre_train, y_train_pred_pls, rowvar=False)
print('R:', "{:.3f}".format(R_Train_pls[0][1]))
R_Squared_Train_pls = r2_score(self.y_pre_train, y_train_pred_pls)
print('R^2:', "{:.3f}".format(R_Squared_Train_pls))
RMSE_Train_pls = math.sqrt(mean_squared_error(self.y_pre_train, y_train_pred_pls))
print('RMSE :', "{:.3f}".format(RMSE_Train_pls))
MAE_Train_pls = mean_absolute_error(self.y_pre_train, y_train_pred_pls)
print('MAE:', "{:.3f}".format(MAE_Train_pls))
# R, R_Squared, RMSE
print('--------------- TEST--------------------')
R_Test_pls = np.corrcoef(self.y_pre_test, y_test_pred_pls, rowvar=False)
print('R:', "{:.3f}".format(R_Test_pls[0][1]))
R_Squared_Test_pls = r2_score(self.y_pre_test, y_test_pred_pls)
print('R^2:', "{:.3f}".format(R_Squared_Test_pls))
RMSE_Test_pls = math.sqrt(mean_squared_error(self.y_pre_test, y_test_pred_pls))
print('RMSE :', "{:.3f}".format(RMSE_Test_pls))
MAE_Test_pls = mean_absolute_error(self.y_pre_test, y_test_pred_pls)
print('MAE:', "{:.3f}".format(MAE_Test_pls))
print('--------------- RPD--------------------')
RPD_Test_pls = np.std(self.y_pre_test) / RMSE_Test_pls
print('RPD:', "{:.2f}".format(RPD_Test_pls))


def load_spectrum():
    plt.scatter(self.y_pre_test, y_test_pred_pls)
    plt.xlabel('Actual Response')
    plt.ylabel('Predicted Response')
    plt.title(f'{name_model_pls} Regression (R²={R_Squared_Test_pls:.2f})')
    reg = np.polyfit(self.y_pre_test, y_test_pred_pls, deg=1)
    trend = np.polyval(reg, self.y_pre_test)
    plt.plot(self.y_pre_test, trend, 'r')
    plt.show()


load_spectrum()