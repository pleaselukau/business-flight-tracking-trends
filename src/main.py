import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    """
    Load data from CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess data (e.g., handle missing values, feature scaling).
    """
    # Drop irrelevant columns
    columns_to_drop = ['Passenger ID', 'First Name', 'Last Name', 'Nationality', 
                       'Airport Name', 'Airport Country Code', 'Country Name', 
                       'Departure Date', 'Arrival Airport', 'Pilot Name']
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Encode relevant categorical columns
    label_encoders = {}
    columns_to_encode = ['Gender', 'Airport Continent', 'Continents', 'Flight Status']
    for column in columns_to_encode:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Feature scaling
    scaler = StandardScaler()
    features_to_scale = ['Age']  # Numerical features that need scaling
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    return data

def train_model(X_train, y_train):
    """
    Train machine learning model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def main():
    # Load data
    data = load_data(r"C:\Users\pleas\OneDrive\Documents\SPRING 2024\CS 4150\business-flight-tracking-trends\data\airline_dataset.csv")
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Split data into features and target variable
    X = data.drop(columns=['Flight Status'])  # Drop the target variable
    y = data['Flight Status']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    main()
