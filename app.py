import os
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# 📌 Function to Load the Model
# ----------------------------
@st.cache_resource  # Caches the model to improve app performance
def load_model():
    model_path = "Model/cnn_lstm_model.h5"  # Ensure this is the correct path
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

# Load Model
model = load_model()
st.success("✅ Model loaded successfully!")

# ----------------------------
# 📂 Load Dataset
# ----------------------------
df_path = "Final_Data/final_apnea_dataset.csv"  # Ensure this is the correct path
if not os.path.exists(df_path):
    st.error(f"❌ Dataset not found: {df_path}")
    raise FileNotFoundError(f"Dataset not found: {df_path}")

df = pd.read_csv(df_path)
st.success("✅ Dataset loaded successfully!")

# ----------------------------
# 🎯 Prepare Features & Labels
# ----------------------------
expected_columns = ["mean_stage", "std_stage", "stage_transitions", "total_apnea_events", "mean_apnea_duration"]
if not all(col in df.columns for col in expected_columns):
    st.error("⚠️ Column mismatch! Please upload a dataset with correct features.")
    st.write("✅ Expected Columns:", expected_columns)
    st.write("📌 Found Columns:", list(df.columns))
    st.stop()

X = df[expected_columns]  # Features
y_true = df["Predicted_Apnea"]  # Labels

# Normalize Features
X = (X - X.mean()) / X.std()

# ----------------------------
# 📊 Streamlit UI
# ----------------------------
st.title("📊 Sleep Apnea Detection App 💤")
st.write("Upload your sleep study CSV file to predict apnea events.")

# File Upload
uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Check column compatibility
    if not all(col in uploaded_df.columns for col in expected_columns):
        st.error("⚠️ Column mismatch! Please upload a valid dataset with correct features.")
        st.write("✅ Expected Columns:", expected_columns)
        st.write("📌 Found Columns:", list(uploaded_df.columns))
    else:
        # Normalize uploaded data
        uploaded_X = (uploaded_df[expected_columns] - X.mean()) / X.std()

        # Make Predictions
        predictions = model.predict(uploaded_X)
        predicted_classes = (predictions > 0.5).astype(int)

        # Display Predictions
        uploaded_df["Predicted_Apnea"] = predicted_classes
        st.write("### 📝 Predictions:")
        st.write(uploaded_df)

        # 📌 Plot Apnea Distribution
        st.subheader("📊 Apnea Event Distribution")
        fig, ax = plt.subplots()
        unique_labels, label_counts = np.unique(predicted_classes, return_counts=True)
        ax.bar(["No Apnea", "Apnea"], label_counts)
        ax.set_ylabel("Count")
        st.pyplot(fig)