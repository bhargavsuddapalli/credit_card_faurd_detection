import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

data = pd.read_csv("data/creditcard.csv")

print(data.head())
print(data.shape)
print(data.info())
print("\nClass Distribution:")
print(data["Class"].value_counts())
print("\nMissing Values:")
print(data.isnull().sum())
# Separate input features and target
X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution Before SMOTE:")
print(y_train.value_counts())

print("\nClass Distribution After SMOTE:")
print(y_train_smote.value_counts())
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train the model
print("\nTraining Random Forest Model...")
model.fit(X_train_smote, y_train_smote)

print("Model training completed successfully!")
# Make predictions
y_pred = model.predict(X_test)

print("\nPrediction completed!")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Calculate model performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model performance metrics
performance = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

performance.to_csv("results/model_performance.csv", index=False)

print("\nModel performance saved successfully!")

# Save model performance metrics
performance = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

performance.to_csv("results/model_performance.csv", index=False)

print("\nModel performance saved successfully!")
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Credit Card Fraud Detection")

plt.show()
import joblib

joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")
# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
# Feature Importance Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features for Credit Card Fraud Detection")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("results/feature_importance.png")

plt.show()
# Model Performance Comparison: Before and After SMOTE

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Before SMOTE": [0.9995, 0.91, 0.79, 0.84],
    "After SMOTE": [0.9994, 0.82, 0.84, 0.83]
})

print("\nModel Comparison:")
print(comparison)
comparison.plot(
    x="Metric",
    y=["Before SMOTE", "After SMOTE"],
    kind="bar",
    figsize=(10, 6)
)

plt.title("Random Forest Performance: Before vs After SMOTE")
plt.ylabel("Score")
plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig("results/smote_comparison.png")

plt.show()
# Save the final SMOTE Random Forest model
joblib.dump(model, "models/random_forest_smote_model.pkl")

print("\nSMOTE Random Forest model saved successfully!")