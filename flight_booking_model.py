
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function to generate fake flight data
def generate_flight_data(num_flights=200):
    random.seed(42)
    np.random.seed(42)
    
    # Define constants
    flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]  # Flight IDs
    start_date = datetime.now().date()
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(15)]  # Upcoming 15 days
    
    # Generate data
    data = []
    for flight_id in flight_ids:
        for date in dates:
            day_of_week = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            total_seats = random.choice([150, 180, 200, 250])
            booked_seats = random.randint(50, total_seats - 10)  # Ensure at least 10 empty seats
            available_seats = total_seats - booked_seats
            holiday_flag = random.choice([0, 1])  # Randomly assign holiday flag
            
            data.append({
                "Flight_ID": flight_id,
                "Date": date,
                "Day_of_Week": day_of_week,
                "Total_Seats": total_seats,
                "Booked_Seats": booked_seats,
                "Available_Seats": available_seats,
                "Holiday_Flag": holiday_flag
            })
            
    return pd.DataFrame(data)

# Generate the dataset
fake_flight_data = generate_flight_data(num_flights=50)

# Save the dataset to CSV
fake_flight_data.to_csv("flight_seat_data.csv", index=False)

# Preprocess the data
print("Fake Flight Seat Dataset:
")
print(fake_flight_data.head())

# Feature encoding and selection
fake_flight_data["Day_of_Week"] = fake_flight_data["Day_of_Week"].astype('category').cat.codes  # Encode days of the week
features = ["Total_Seats", "Holiday_Flag", "Day_of_Week"]  # Removed Booked Seats
booked_target = "Booked_Seats"

X = fake_flight_data[features]
y = fake_flight_data[booked_target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model to predict Booked Seats
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Absolute Error for Booked Seats: {mae:.2f}")

# Function to predict available seats based on user input
def predict_available_seats(date, total_seats, holiday_flag):
    day_of_week = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    day_of_week_encoded = pd.Series([day_of_week]).astype('category').cat.codes[0]  # Encode input day
    
    # Create input features
    input_features = pd.DataFrame({
        "Total_Seats": [total_seats],
        "Holiday_Flag": [holiday_flag],
        "Day_of_Week": [day_of_week_encoded]
    })
    
    # Predict booked seats
    predicted_booked_seats = model.predict(input_features)[0]
    available_seats = total_seats - predicted_booked_seats
    return max(0, available_seats)  # Ensure no negative available seats

# User input example
print("\n--- Predict Available Seats for a Given Date ---")
user_date = input("Enter date (YYYY-MM-DD): ")
user_total_seats = int(input("Enter total seats: "))
user_holiday_flag = int(input("Is it a holiday? (1 for Yes, 0 for No): "))

predicted_available_seats = predict_available_seats(user_date, user_total_seats, user_holiday_flag)
print(f"Predicted Available Seats: {predicted_available_seats:.2f}")
