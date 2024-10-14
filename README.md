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

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Superstore dataset
# Replace the path with the location where the Kaggle dataset is stored
data = pd.read_csv('/content/Super_Store_data.csv', parse_dates=['Order Date'], index_col='Order Date',encoding='ISO-8859-1')

# Filter for 'Furniture' sales
furniture_sales = data[data['Category'] == 'Furniture']

# Resample data to a monthly frequency (sum of sales per month)
monthly_sales = furniture_sales['Sales'].resample('MS').sum()

# Scaling the Data using MinMaxScaler
scaler = MinMaxScaler()
sales_scaled = pd.Series(scaler.fit_transform(monthly_sales.values.reshape(-1, 1)).flatten(), index=monthly_sales.index)

# Split into training and testing sets (80% train, 20% test)
train_data = sales_scaled[:int(len(sales_scaled) * 0.8)]
test_data = sales_scaled[int(len(sales_scaled) * 0.8):]

# Step 2: Fit the Holt-Winters model with additive trend and seasonality
fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Step 3: Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions for Furniture Sales')
plt.legend(loc='best')
plt.show()

# Step 4: Fit the final model on the entire dataset
# Final model with additive trend and seasonality
final_model = ExponentialSmoothing(
    sales_scaled, trend='add', seasonal='add', seasonal_periods=12
).fit()


# Forecast future values (next 12 months)
forecast_predictions = final_model.forecast(steps=12)

# Step 5: Plot the actual and forecasted sales
plt.figure(figsize=(12, 8))
monthly_sales.plot(legend=True, label='Current Furniture Sales')
forecast_predictions.plot(legend=True, label='Forecasted Furniture Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Furniture Sales Forecast')
plt.show()

```

### OUTPUT:

![Screenshot 2024-10-14 103902](https://github.com/user-attachments/assets/4d418290-8495-4625-8554-9f46e62c5a3a)


#### TEST_PREDICTION
![Screenshot 2024-10-14 103721](https://github.com/user-attachments/assets/df5f8c6e-0169-436a-89ec-3a3a22ab7c19)

#### FINAL_PREDICTION
![Screenshot 2024-10-14 103732](https://github.com/user-attachments/assets/b8081a78-d3f2-4a3c-a599-7604c4bbfcd1)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
