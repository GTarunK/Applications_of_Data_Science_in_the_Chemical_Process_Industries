import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the data into a pandas DataFrame
data = pd.DataFrame({
    'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
    'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
    'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
    'Y2': [3.9, 3.8, 3.6, 3.5, 4, 3.8, 3.7, 3.6, 6.9, 6.6, 6.1, 5.8, 7.9, 7.6, 7.1, 6.5, 3.8, 3.6, 5.3, 4.7, 6.1, 3.5, 3.3, 3.3]
})

# Split the data into input features (X) and target variable (Y)
X = data.drop('Y2', axis=1)
Y = data['Y2']

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, Y)

# Print the coefficients of the model
print('Coefficients:', model.coef_)

# Generate predictions on the training set
Y_pred = model.predict(X_poly)

# Plot the actual vs predicted values
plt.scatter(Y, Y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title("Actual Y2 vs Predicted Y2 (III Poly. Model)")
plt.show()

# Print the R-squared score of the model
print('R-squared:', model.score(X_poly, Y))

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Calculate percentage error
pct_error = np.abs((Y - Y_pred) / Y) * 100
avg_pct_error = np.mean(pct_error)
print('Percentage Error:', avg_pct_error)

# Calculate maximum error
max_error = np.max(np.abs(Y - Y_pred))
print('Maximum Error:', max_error)

# Calculate R-squared value
r_squared = r2_score(Y, Y_pred)
print('R-squared:', r_squared)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print('RMSE:', rmse)

# Calculate MAE
mae = mean_absolute_error(Y, Y_pred)
print('MAE:', mae)

# Calculate MAPE
mape = np.mean(pct_error)
print('MAPE:', mape)

# Calculate AIC
n = len(Y)
k = 3
rss = np.sum((Y - Y_pred) ** 2)
aic = n * np.log(rss / n) + 2 * k
print('AIC:', aic)

# Calculate BIC
bic = n * np.log(rss / n) + k * np.log(n)
print('BIC:', bic)

