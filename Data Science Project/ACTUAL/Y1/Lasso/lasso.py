import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data into a Pandas dataframe
data = pd.DataFrame({'X1': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 17, 20],
                     'X2': [90, 90, 90, 90, 110, 110, 110, 110, 130, 130, 130, 130, 150, 150, 150, 150, 100, 100, 120, 120, 140, 110, 110, 110],
                     'X3': [0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.15, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2],
                     'Y1': [37, 62, 67, 78.1, 44, 67, 76, 84.9, 43, 59, 72, 78, 56, 66, 74, 81.9, 71, 82.4, 74, 81.5, 79.5, 86.2, 86.6, 87]})

# Separate the input variables and the target variable
X = data[['X1', 'X2', 'X3']]
Y = data['Y1']

# Create the Lasso regression model
model = Lasso(alpha=0.1)

# Fit the model to the data
model.fit(X, Y)

# Make predictions using the model
Y_pred = model.predict(X)

# Print the coefficients of the model
print('Coefficients:', model.coef_)

# Print the intercept of the model
print('Intercept:', model.intercept_)

# Print the mathematical equation of the model
print('Mathematical equation: Y1 =', model.intercept_, '+', model.coef_[0], '* X1', '+', model.coef_[1], '* X2', '+', model.coef_[2], '* X3')

# Calculate metrics
r_squared = r2_score(Y, Y_pred)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
mae = mean_absolute_error(Y, Y_pred)
mape = np.mean(np.abs(Y - Y_pred) / Y) * 100
percentage_error = np.mean(np.abs(Y - Y_pred) / Y) * 100
max_error = np.max(np.abs(Y - Y_pred))
n = len(Y)
p = X.shape[1]
aic = n * np.log(mean_squared_error(Y, Y_pred)) + 2 * p
bic = n * np.log(mean_squared_error(Y, Y_pred)) + np.log(n) * p

# Print metrics
print("R-squared:", r_squared)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
print("Percentage Error:", percentage_error)
print("Max Error:", max_error)
print("AIC:", aic)
print("BIC:", bic)

# Plot actual vs predicted values
plt.scatter(Y, Y_pred)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Y1 vs Predicted Y1 (Lasso Model)")
plt.show()