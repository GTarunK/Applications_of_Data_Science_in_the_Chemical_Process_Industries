import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
# X = pd.DataFrame({'X1': [10]*20 + [13, 17, 20]*2, 
#                   'X2': [90]*4 + [110]*4 + [130]*4 + [150]*4 + [100, 120]*2 + [140]*3 + [110]*4,
#                   'X3': [0.05, 0.1, 0.15, 0.2]*6})
# Y = pd.DataFrame({'Y1': [37, 62, 67, 78.1, 44, 67, 76, 84.9, 43, 59, 72, 78, 56, 66, 74, 81.9,
#                          71, 82.4, 74, 81.5, 79.5, 86.2, 86.6, 87]})

df = pd.DataFrame({'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
                   'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
                   'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
                   'Y1': [37, 62, 67, 78.1, 44, 67, 76, 84.9, 43, 59, 72, 78, 56, 66, 74, 81.9, 71, 82.4, 74, 81.5, 79.5, 86.2, 86.6, 87]})

# create the X and Y arrays
X = df[['X1', 'X2', 'X3']]
Y = df['Y1']

# Fit linear regression model
model = LinearRegression()
model.fit(X, Y)

# Print model coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# Predict Y1 values using the model
Y_pred = model.predict(X)

# Generate actual vs predicted values plot
plt.scatter(Y, Y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('Actual Y1')
plt.ylabel('Predicted Y1')
plt.title("Actual Y1 vs Predicted Y1 (Linear Model)")
plt.show()

# evaluate the model using various metrics
r2 = r2_score(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y, Y_pred)
mape = np.mean(np.abs((Y - Y_pred) / Y)) * 100
percentage_error = np.mean(np.abs((Y - Y_pred) / Y)) * 100
max_error = np.max(np.abs(Y - Y_pred))
aic = len(Y) * np.log(mse) + 2 * (3)
bic = len(Y) * np.log(mse) + np.log(len(Y)) * (3)

# print the metrics
print('R-squared:', r2)
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE:', mape)
print('Percentage Error:', percentage_error)
print('MSE:', mse)
print('Maximum Error:', max_error)
print('AIC:', aic)
print('BIC:', bic)
