import pandas as pd
import joblib

# Load trained SMOTE Random Forest model
model = joblib.load("models/random_forest_smote_model.pkl")

# Load dataset
data = pd.read_csv("data/creditcard.csv")

print("Model loaded successfully!")

# Ask user for transaction row number
row_number = int(input("Enter transaction row number (0 to 284806): "))

# Get transaction features
sample = data.drop("Class", axis=1).iloc[[row_number]]

# Get actual class
actual_class = data["Class"].iloc[row_number]

# Predict transaction
prediction = model.predict(sample)

# Get prediction probability
probability = model.predict_proba(sample)

print("\n-------------------------------")
print("Actual Class:", actual_class)

if prediction[0] == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: NORMAL TRANSACTION")

print("Normal Probability:", probability[0][0])
print("Fraud Probability:", probability[0][1])
print("-------------------------------")