 # Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data file using relative paths
data_file_path = os.path.join(current_dir, '..', 'data', 'Flight_data.csv')

# Load the dataset
flight_data = pd.read_csv(data_file_path)

# Feature analysis
# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
flight_data['Departure City'] = label_encoder.fit_transform(flight_data['Departure City'])
flight_data['Arrival City'] = label_encoder.fit_transform(flight_data['Arrival City'])
flight_data['Customer ID'] = label_encoder.fit_transform(flight_data['Customer ID'])
flight_data['Name'] = label_encoder.fit_transform(flight_data['Name'])
flight_data['Booking Class'] = label_encoder.fit_transform(flight_data['Booking Class'])
flight_data['Frequent Flyer Status'] = label_encoder.fit_transform(flight_data['Frequent Flyer Status'])
flight_data['Route'] = label_encoder.fit_transform(flight_data['Route'])
flight_data['Origin'] = label_encoder.fit_transform(flight_data['Origin'])
flight_data['Destination'] = label_encoder.fit_transform(flight_data['Destination'])
flight_data['Churned'] = label_encoder.fit_transform(flight_data['Churned'])

# Convert Departure Date to datetime
flight_data['Departure Date'] = pd.to_datetime(flight_data['Departure Date'])

# Create new features for month and year from Departure Date
flight_data['Departure Month'] = flight_data['Departure Date'].dt.month
flight_data['Departure Year'] = flight_data['Departure Date'].dt.year

# Drop irrelevant columns
flight_data.drop(['Departure Date'], axis=1, inplace=True)

# Split dataset into features and target variable
X = flight_data.drop(['Profitability'], axis=1)
y = flight_data['Profitability']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = rf_regressor.predict(X_train_scaled)
y_pred_test = rf_regressor.predict(X_test_scaled)

# Calculate RMSE
rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_regressor.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Ask the user to input data for a new flight
print("Enter details for the new flight:")
departure_city = input("Departure City: ")
arrival_city = input("Arrival City: ")
flight_duration = float(input("Flight Duration (hours): "))
delay_minutes = float(input("Delay Minutes: "))
customer_id = int(input("Customer ID: "))
name = input("Name: ")
booking_class = input("Booking Class: ")
frequent_flyer_status = input("Frequent Flyer Status: ")
route = input("Route: ")
ticket_price = float(input("Ticket Price: "))
competitor_price = float(input("Competitor Price: "))
demand = int(input("Demand: "))
origin = input("Origin: ")
destination = input("Destination: ")
loyalty_points = int(input("Loyalty Points: "))
churned = int(input("Churned (0 for No, 1 for Yes): "))
departure_month = int(input("Departure Month: "))
departure_year = int(input("Departure Year: "))

# Create a DataFrame for the new flight data
new_data = pd.DataFrame({
    'Departure City': [departure_city],
    'Arrival City': [arrival_city],
    'Flight Duration': [flight_duration],
    'Delay Minutes': [delay_minutes],
    'Customer ID': [customer_id],
    'Name': [name],
    'Booking Class': [booking_class],
    'Frequent Flyer Status': [frequent_flyer_status],
    'Route': [route],
    'Ticket Price': [ticket_price],
    'Competitor Price': [competitor_price],
    'Demand': [demand],
    'Origin': [origin],
    'Destination': [destination],
    'Loyalty Points': [loyalty_points],
    'Churned': [churned],
    'Departure Month': [departure_month],
    'Departure Year': [departure_year]
})

# Convert categorical variables to numerical using Label Encoding
new_data['Departure City'] = label_encoder.transform(new_data['Departure City'])
new_data['Arrival City'] = label_encoder.transform(new_data['Arrival City'])
new_data['Name'] = label_encoder.transform(new_data['Name'])
new_data['Booking Class'] = label_encoder.transform(new_data['Booking Class'])
new_data['Frequent Flyer Status'] = label_encoder.transform(new_data['Frequent Flyer Status'])
new_data['Route'] = label_encoder.transform(new_data['Route'])
new_data['Origin'] = label_encoder.transform(new_data['Origin'])
new_data['Destination'] = label_encoder.transform(new_data['Destination'])
new_data['Churned'] = label_encoder.transform(new_data['Churned'])

# Scale the features
new_data_scaled = scaler.transform(new_data)

# Calculate the expected profitability based on the training data
expected_profitability = y_train.mean()
print("\nExpected Profitability (based on training data):", expected_profitability)

# Predict the profitability for the new flight data
predicted_profitability = rf_regressor.predict(new_data_scaled)[0]
print("Predicted Profitability for the new flight:", predicted_profitability)

