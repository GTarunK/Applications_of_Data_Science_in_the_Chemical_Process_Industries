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
                     'Y2': [3.9, 3.8, 3.6, 3.5, 4, 3.8, 3.7, 3.6, 6.9, 6.6, 6.1, 5.8, 7.9, 7.6, 7.1, 6.5, 3.8, 3.6, 5.3, 4.7, 6.1, 3.5, 3.3, 3.3]})

# Separate the input variables and the target variable
X = data[['X1', 'X2', 'X3']]
Y = data['Y2']

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
print('Mathematical equation: Y2 =', model.intercept_, '+', model.coef_[0], '* X1', '+', model.coef_[1], '* X2', '+', model.coef_[2], '* X3')

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
plt.title("Actual Y2 vs Predicted Y2 (Lasso Model)")
plt.show()