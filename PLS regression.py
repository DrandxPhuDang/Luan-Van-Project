import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from kennard_stone import train_test_split, KFold
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# split data set
url = f'D:\Luan Van Tot Nghiep\Data\Final_Data\Data_ALL.csv'
df = pd.read_csv(url)
listWavelegh = df.iloc[:0, 9:]
folder_bc = r'D:\Luan Van Tot Nghiep\Data\loc_waves\change_val_file_waves.csv'
listW = pd.read_csv(folder_bc)
features = [f'{e}' for e in listWavelegh]
X = df[features]
y = df['Brix']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a PLSRegression object with the desired number of components
pls = PLSRegression(n_components=2)

# Fit the model to your training data
pls.fit(X_train, y_train)

# Predict the response variable using the trained model
y_test_pred = pls.predict(X_test)

# Calculate the R² score of the model on the test data
score = pls.score(X_test, y_test)
print(f'R²: {score}')
print(f"Accuracy: {score * 100:.3f}%")
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")

# Plot the predicted values against the actual values
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Response')
plt.ylabel('Predicted Response')
plt.title(f'PLS Regression (R²={score:.2f})')
reg = np.polyfit(y_test, y_test_pred, deg=1)
trend = np.polyval(reg, y_test)
plt.plot(y_test, trend, 'r')
plt.show()
