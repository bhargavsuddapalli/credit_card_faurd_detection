import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load trained model
model = joblib.load("models/random_forest_smote_model.pkl")

# Load dataset
data = pd.read_csv("data/creditcard.csv")


# Title
st.title("💳 Credit Card Fraud Detection Using Random Forest")
st.write(
    "This application predicts whether a credit card transaction is "
    "Normal or Fraudulent using a Random Forest machine learning model."
)

st.divider()


# ---------------- MODEL PERFORMANCE ----------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "99.94%")

with col2:
    st.metric("Precision", "82%")

with col3:
    st.metric("Recall", "84%")

with col4:
    st.metric("F1-Score", "83%")


st.divider()


# ---------------- TRANSACTION PREDICTION ----------------

st.subheader("🔍 Transaction Prediction")

row_number = st.number_input(
    "Enter Transaction Row Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)


if st.button("Predict Transaction"):

    # Get transaction features
    sample = data.drop("Class", axis=1).iloc[[row_number]]

    # Actual class
    actual_class = data["Class"].iloc[row_number]

    # Prediction
    prediction = model.predict(sample)

    # Prediction probability
    probability = model.predict_proba(sample)

    st.divider()

    st.subheader("Prediction Result")

    # Display prediction
    if prediction[0] == 1:
        st.error("⚠️ FRAUDULENT TRANSACTION DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    # Probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Probability",
            f"{probability[0][0] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Fraud Probability",
            f"{probability[0][1] * 100:.2f}%"
        )

    # Actual class
    st.subheader("Actual Class")

    if actual_class == 1:
        st.write("⚠️ Fraudulent Transaction")
    else:
        st.write("✅ Normal Transaction")


st.divider()


# ---------------- PROJECT VISUALIZATIONS ----------------

st.subheader("📈 Project Visualizations")

tab1, tab2 = st.tabs(
    ["Feature Importance", "SMOTE Comparison"]
)

with tab1:
    st.image(
        "results/feature_importance.png",
        caption="Feature Importance in Random Forest Model"
    )

with tab2:
    st.image(
        "results/smote_comparison.png",
        caption="Model Performance Before and After SMOTE"
    )


st.divider()

st.caption(
    "Credit Card Fraud Detection Project | "
    "Machine Learning using Random Forest and SMOTE"
)