#this one is a regressor model so i tested it for price pred instead on same dataset
# Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data file using relative paths
data_file_path = os.path.join(current_dir, '..', 'data', 'Flight_data.csv')

data = pd.read_csv(data_file_path)

# Encoding categorical variables
encoder = LabelEncoder()
data['Departure City'] = encoder.fit_transform(data['Departure City'])
data['Arrival City'] = encoder.fit_transform(data['Arrival City'])
data['Name'] = encoder.fit_transform(data['Name'])
data['Booking Class'] = encoder.fit_transform(data['Booking Class'])
data['Frequent Flyer Status'] = encoder.fit_transform(data['Frequent Flyer Status'])
data['Route'] = encoder.fit_transform(data['Route'])
data['Origin'] = encoder.fit_transform(data['Origin'])
data['Destination'] = encoder.fit_transform(data['Destination'])
data['Churned'] = encoder.fit_transform(data['Churned'])

# Convert 'Departure Date' to datetime
data['Departure Date'] = pd.to_datetime(data['Departure Date'])

# Extract month and year from 'Departure Date'
data['Departure Month'] = data['Departure Date'].dt.month
data['Departure Year'] = data['Departure Date'].dt.year

# Drop unnecessary columns
data.drop(['Departure Date', 'Customer ID'], axis=1, inplace=True)

# Split data into features and target variable
X = data.drop('Ticket Price', axis=1)
y = data['Ticket Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Interpret the Mean Squared Error and print predictions
print("The Mean Squared Error indicates the average squared difference between the actual ticket prices and the predicted ticket prices by the model.")

# Example of printing predictions
print("Example of Predictions:")
for i in range(5):  # Print predictions for the first 5 samples
    print(f"For the {i+1}st sample, the predicted ticket price is ${y_pred[i]:.2f}.")
