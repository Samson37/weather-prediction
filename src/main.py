import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import load_data, preprocess_data
from models import create_model

def main():
    # Load data
    data = load_data('data/weather_data.csv')

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the model
    model = create_model("random_forest")  # Choose your desired model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

if __name__ == "__main__":
    main()