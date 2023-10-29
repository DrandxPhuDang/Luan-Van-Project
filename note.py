import math
import optuna
import numpy as np
import pandas as pd
import tensorflow
from livelossplot import PlotLossesKerasTF
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from kennard_stone import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint

'''get data'''
df = pd.read_csv(r"D:\Luan Van\Data\Final_Data\Final.csv")
list_features = df.iloc[:0, 12:]
features = [f'{e}' for e in list_features]
X = df[features].values
y = df['Brix'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


epochs = 500
batch = 256
lr = 0.0006
tensorflow.keras.backend.clear_session()

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
LR = 0.01*batch/256.
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=LR), loss='mse', metrics=['mse'])
model.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50,
                        verbose=1, mode='auto', restore_best_weights=True)

# cnn_history = model.fit(X_train, y_train, callbacks=[monitor],
#                         validation_data=(X_test, y_test), epochs=epochs, verbose=2)

early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, mode='auto', restore_best_weights=True)
rdlr = ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=0)
MODEL_NAME = 'base_regression_model.h5'
checkpointer = ModelCheckpoint(filepath=MODEL_NAME, verbose=1, save_best_only=True)
# plot_losses = PlotLossesKerasTF()

model.fit(X_train, y_train, batch_size=batch, epochs=epochs,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, rdlr, early_stop], verbose=2)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


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
print_score(y_train, y_pred_train)
print('--------------- TEST--------------------')
score_test = print_score(y_test, y_pred_test)
print('--------------- RPD--------------------')
RPD_Test = np.std(y_test) / score_test[2]
print('RPD:', "{:.2f}".format(RPD_Test))
