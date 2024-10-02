import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Replace 'your_data.csv' with the actual path to your CSV file
data = pd.read_csv("olddata.csv")

# Extract features and labels
X = data.iloc[:, 0].values.reshape(-1, 1)  # Assuming the timestamp is the feature
y = data.iloc[:, 1].values

# Check for string values in X and convert to numeric if necessary
if pd.api.types.is_string_dtype(X):
    X = pd.to_numeric(X, errors='coerce').fillna(-1)  # Replace non-numeric values with -1

# Encode labels if categorical
if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=1))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])



def main():
    st.title("Prediction App")

    # Input for new data
    new_data = st.number_input("Enter new data")

    if st.button("Predict"):
        new_data = np.array([new_data]).reshape(-1, 1)
        prediction = model.predict(new_data)
        st.success(f"Prediction: {prediction[0][0]}")

if __name__ == "__main__":
    main()


