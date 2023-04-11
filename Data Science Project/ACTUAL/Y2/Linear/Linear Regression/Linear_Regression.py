import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.DataFrame({'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
                   'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
                   'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
                   'Y2': [3.9, 3.8, 3.6, 3.5, 4, 3.8, 3.7, 3.6, 6.9, 6.6, 6.1, 5.8, 7.9, 7.6, 7.1, 6.5, 3.8, 3.6, 5.3, 4.7, 6.1, 3.5, 3.3, 3.3]})

# create the X and Y arrays
X = df[['X1', 'X2', 'X3']]
Y = df['Y2']

# Fit linear regression model
model = LinearRegression()
model.fit(X, Y)

# Print model coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# Predict Y2 values using the model
Y_pred = model.predict(X)

# Generate actual vs predicted values plot
plt.scatter(Y, Y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('Actual Y2')
plt.ylabel('Predicted Y2')
plt.title("Actual Y2 vs Predicted Y2 (Linear Model)")
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

