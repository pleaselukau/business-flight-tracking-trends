# Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Drop unnecessary columns and datetime columns
data.drop(['Departure Date', 'Customer ID'], axis=1, inplace=True)

# Define the target variable: Rank of cities based on business activity
# For example, we can create classes based on the number of business activities in each city
# for starter, i have only created 3 classes: High, Medium, Low
# we need to define a logic for ranking the cities based on business requirements
def rank_cities(row, X_train):
    departure_city = row['Departure City']
    arrival_city = row['Arrival City']
    
    # Example : Count the number of flights departing from and arriving at each city
    departure_flights = X_train[X_train['Departure City'] == departure_city].shape[0]
    arrival_flights = X_train[X_train['Arrival City'] == arrival_city].shape[0]
    
    total_flights = departure_flights + arrival_flights
    
    # Example : Rank cities based on the total number of flights
    if total_flights >= 1000:
        return 'High'
    elif 500 <= total_flights < 1000:
        return 'Medium'
    else:
        return 'Low'

# Split data into features and target variable
X = data.drop(['Ticket Price'], axis=1)
y = data.apply(lambda row: rank_cities(row, X), axis=1)

# Convert target variable to categorical
y = pd.Categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example of printing predictions for business trends and city ranks
print("Example of Predictions:")
for i in range(5):  # Print predictions for the first 5 samples
    print(f"For the {i+1}st sample, the predicted city rank is '{y_pred[i]}'.")
