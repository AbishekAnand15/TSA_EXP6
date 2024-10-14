### Name  : ABISHEK XAVIER A
### Reg.No: 212222230004
### Date  : 

# Ex.No: 6               HOLT WINTERS METHOD
### AIM:
   To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. Load and resample the gold price data to monthly frequency, selecting the 'Price' column.
2. Scale the data using Minmaxscaler then split into training (80%) and testing (20%) sets.
3. Fit an additive Holt-Winters model to the training data and forecast on the test data.
4. Evaluate model performance using MAE and RMSE, and plot the train, test, and prediction results.
5. Train a final multiplicative Holt-Winters model on the full dataset and forecast future gold prices.
### PROGRAM:
```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

data = pd.read_csv('/content/Gold Price.csv', index_col='Date', parse_dates=True)

# Calculate the mean of all columns resampled to month start frequency
data = data.resample('MS').mean() 

# Select the 'PRICE' column for analysis
data = data['Price']

# Scaling the Data using MinMaxScaler 
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

final_model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Forecast future values
forecast_predictions = final_model.forecast(steps=12)

data.plot(figsize=(12, 8), legend=True, label='Current Gold Price')
forecast_predictions.plot(legend=True, label='Forecasted Gold Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold Price Forecast')
plt.show()
```

### OUTPUT:

![image](https://github.com/user-attachments/assets/d5e9e423-badf-43e8-a20a-f11dbaf47d2d)


#### TEST_PREDICTION
![Untitled](https://github.com/user-attachments/assets/babcd073-db02-49c6-889b-1e67e042bb0c)

#### FINAL_PREDICTION
![Untitled](https://github.com/user-attachments/assets/49696fd9-e49a-4436-83de-07b43db8ed31)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
