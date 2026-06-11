# weather data analysis and temperature prediction
# using linear regression on historical weather data

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
# download from: https://www.kaggle.com/datasets/muthuj7/weather-dataset
# save as 'weatherHistory.csv' in the same folder

df = pd.read_csv('weatherHistory.csv')

print("shape of dataset:", df.shape)
print("\ncolumns:", df.columns.tolist())
print("\nfirst 5 rows:")
print(df.head())

# check for missing values
print("\nmissing values:")
print(df.isnull().sum())

# keep useful columns only
df = df[['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']]
df.columns = ['date', 'temperature', 'humidity', 'wind_speed', 'pressure']

# drop rows with missing values if any
df.dropna(inplace=True)

# basic stats
print("\nbasic statistics:")
print(df.describe())

# plot temperature over time (first 200 rows so its easy to see)
plt.figure(figsize=(10, 4))
plt.plot(df['temperature'][:200].values, color='orangered')
plt.title('Temperature Over Time (first 200 records)')
plt.xlabel('Record Number')
plt.ylabel('Temperature (C)')
plt.tight_layout()
plt.savefig('temperature_trend.png')
plt.show()
print("temperature trend chart saved")

# correlation heatmap
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(df[['temperature', 'humidity', 'wind_speed', 'pressure']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("correlation heatmap saved")

# scatter plot - humidity vs temperature
plt.figure(figsize=(6, 4))
plt.scatter(df['humidity'], df['temperature'], alpha=0.3, color='steelblue', s=5)
plt.title('Humidity vs Temperature')
plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')
plt.tight_layout()
plt.savefig('humidity_vs_temp.png')
plt.show()
print("scatter plot saved")

# prepare data for prediction
x = df[['humidity', 'wind_speed', 'pressure']]
y = df['temperature']

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# train linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# predict on test data
y_pred = model.predict(x_test)

# model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test[:100].values, y_pred[:100], alpha=0.6, color='purple', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
print("actual vs predicted chart saved")

# try predicting with custom input
print("\npredicting temperature for custom input:")
custom = pd.DataFrame({'humidity': [0.85], 'wind_speed': [14.0], 'pressure': [1015.0]})
predicted_temp = model.predict(custom)[0]
print(f"humidity=0.85, wind_speed=14, pressure=1015 --> predicted temp: {predicted_temp:.2f} C")