# 🍷 Enhanced Wine Quality Prediction with Better Accuracy

import streamlit as st
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(page_title="Wine Quality Predictor")

st.title("🍷 Wine Quality Prediction")
st.write("Predict if wine is **Bad**, **Average**, or **Good** using its chemical attributes.")

# Load and clean dataset
df = pd.read_csv("winequality-red.csv", sep=";")
df.columns = df.columns.str.strip()

# Convert 'quality' to 3 categories
def quality_to_category(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

df["quality_category"] = df["quality"].apply(quality_to_category)

# Preview
with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head())

# Features and label
X = df.drop(["quality", "quality_category"], axis=1)
y = df["quality_category"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Stratified Split for balanced classes
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Train a better tuned Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"🎯 Model Accuracy: **{accuracy * 100:.2f}%**")

# Report
with st.expander("📈 Model Performance Report"):
    st.text(classification_report(y_test, y_pred, target_names=["Bad", "Average", "Good"]))

# Input Section
st.header("🔍 Predict Wine Quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, step=0.01)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, step=0.1)
chlorides = st.number_input("Chlorides", 0.0, 1.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, step=1.0)
density = st.number_input("Density", 0.9900, 1.0050, step=0.0001, format="%.4f")
pH = st.number_input("pH", 2.5, 4.5, step=0.01)
sulphates = st.number_input("Sulphates", 0.0, 2.0, step=0.01)
alcohol = st.number_input("Alcohol", 8.0, 15.0, step=0.1)

# Create input DataFrame
input_data = pd.DataFrame([{
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
}])

# Standardize input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
labels = {0: "Bad", 1: "Average", 2: "Good"}

# Display
st.subheader("📌 Prediction Result:")
if prediction == 0:
    st.error(f"Wine Quality: {labels[prediction]} 🍷")
elif prediction == 1:
    st.warning(f"Wine Quality: {labels[prediction]} 🍷")
else:
    st.success(f"Wine Quality: {labels[prediction]} 🍷")

with st.expander("🧪 Input Summary"):
    st.dataframe(input_data)
