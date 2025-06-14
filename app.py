
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv("dataset.csv")

# Encode all categorical columns
label_encoders = {}
for column in df.columns[1:]:  # skip 'No'
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split features and target
X = df.drop(columns=["No", "Menang"])
y = df["Menang"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Streamlit App
st.title("Prediksi Kemenangan Tim")

# User Input
user_input = {}
for column in X.columns:
    le = label_encoders[column]
    options = le.classes_
    value = st.selectbox(f"{column}", options)
    user_input[column] = le.transform([value])[0]

# Predict
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    prediction = knn.predict(input_df)[0]
    hasil = label_encoders["Menang"].inverse_transform([prediction])[0]
    st.success(f"Hasil Prediksi: {hasil}")
