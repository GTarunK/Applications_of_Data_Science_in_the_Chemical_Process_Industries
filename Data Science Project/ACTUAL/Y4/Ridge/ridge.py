import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the data into a pandas DataFrame
data = pd.DataFrame({
    'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
    'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
    'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
    'Y4': [18, 25, 31, 42, 22, 29, 43, 52, 21, 27, 36, 44, 19, 22, 32, 41, 34, 45, 40, 50, 42, 83, 153, 171]
})

# Split the data into input features (X) and target variable (Y)
X = data.drop('Y4', axis=1)
y = data['Y4']

# Create a Ridge regression object
ridge = Ridge(alpha=1.0)

# Fit the Ridge regression model
ridge.fit(X, y)

# Extract the coefficients and intercept
coefficients = ridge.coef_
intercept = ridge.intercept_

# Create the equation for the Ridge regression line
equation = 'Y4 = {:.3f} + {:.3f}*X1 + {:.3f}*X2 + {:.3f}*X3'.format(intercept, coefficients[0], coefficients[1], coefficients[2])
print('Ridge Regression Equation:', equation)

# Predict y values using the Ridge regression model
y_pred = ridge.predict(X)

# Calculate the metrics
pe = np.mean(np.abs((y - y_pred) / y)) * 100
me = np.max(np.abs(y - y_pred))
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
n = len(y)
k = 3 # number of independent variables
aic = n*np.log(sum((y - y_pred)**2)/n) + 2*k
bic = n*np.log(sum((y - y_pred)**2)/n) + k*np.log(n)

# Print the metrics
print('Percentage Error: {:.3f}'.format(pe))
print('Maximum Error: {:.3f}'.format(me))
print('R-squared: {:.3f}'.format(r2))
print('RMSE: {:.3f}'.format(rmse))
print('MAE: {:.3f}'.format(mae))
print('MAPE: {:.3f}'.format(mape))
print('AIC: {:.3f}'.format(aic))
print('BIC: {:.3f}'.format(bic))

# Plot the actual vs predicted y values
plt.scatter(y, y_pred)
plt.plot(y, y, color='red')
plt.xlabel('Actual Y Values')
plt.ylabel('Predicted Y Values')
plt.title("Actual Y4 vs Predicted Y4 (Ridge Model)")
plt.show()
