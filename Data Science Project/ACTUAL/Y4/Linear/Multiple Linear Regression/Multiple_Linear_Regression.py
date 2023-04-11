import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# load the data into a Pandas DataFrame
df = pd.DataFrame({
    'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
    'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
    'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
    'Y4': [18, 25, 31, 42, 22, 29, 43, 52, 21, 27, 36, 44, 19, 22, 32, 41, 34, 45, 40, 50, 42, 83, 153, 171]
})

# create the independent variables (X) and dependent variable (y) arrays
X = df[['X1', 'X2', 'X3']].values
y = df['Y4'].values

# create the linear regression model
model = LinearRegression().fit(X, y)

# get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# get the R-squared value
r_squared = r2_score(y, model.predict(X))

# print the coefficients, intercept, and R-squared value
print('Coefficients:', coefficients)
print('Intercept:', intercept)
print('R-squared:', r_squared)

# plot the actual vs predicted values
plt.scatter(y, model.predict(X))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Y values')
plt.ylabel('Predicted Y values')
plt.title("Actual Y4 vs Predicted Y4 (Multiple Linear Model)")
plt.show()

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
pe = np.mean(np.abs((y - y_pred) / y)) * 100 # Percentage Error
me = np.max(np.abs(y - y_pred)) # Maximum Error
r2 = r2_score(y, y_pred) # R-squared
rmse = mean_squared_error(y, y_pred, squared=False) # RMSE
mae = mean_absolute_error(y, y_pred) # MAE
mape = np.mean(np.abs((y - y_pred) / y)) * 100 # MAPE
n = len(y)
# k = len(X.columns)
k = X.shape[1]
aic = n * np.log(mean_squared_error(y, y_pred)) + 2 * k # AIC
bic = n * np.log(mean_squared_error(y, y_pred)) + np.log(n) * k # BIC

# Print the results
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Percentage Error:', pe)
print('Maximum Error:', me)
print('R-squared:', r2)
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE:', mape)
print('AIC:', aic)
print('BIC:', bic)

