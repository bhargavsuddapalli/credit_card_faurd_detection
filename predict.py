import pandas as pd
import joblib

# Load the trained SMOTE Random Forest model
model = joblib.load("models/random_forest_smote_model.pkl")

print("Model loaded successfully!")
# Load dataset
data = pd.read_csv("data/creditcard.csv")

# Select only fraudulent transactions
fraud_data = data[data["Class"] == 1]

# Take the first fraudulent transaction
sample = fraud_data.drop("Class", axis=1).iloc[[0]]

# Actual class
actual_class = fraud_data["Class"].iloc[0]

# Predict transaction
prediction = model.predict(sample)

# Prediction probability
probability = model.predict_proba(sample)

print("\nActual Class:", actual_class)

if prediction[0] == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: NORMAL TRANSACTION")

print("Prediction Probability:", probability)