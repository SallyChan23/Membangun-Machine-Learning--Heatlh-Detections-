import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv("healthcare_dataset.csv")

    X = df.drop(columns=['Medical Condition', 'Name', 'Room Number', 'Date of Admission', 'Discharge Date'])
    y = df['Medical Condition']


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = ['Gender', 'Blood Type', 'Admission Type', 'Insurance Provider', 
                        'Doctor', 'Hospital', 'Medication', 'Test Results']

    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

    numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    X_encoded["target"] = y_encoded

    os.makedirs("preprocessing", exist_ok=True)
    X_encoded.to_csv("preprocessing/healthcare_preprocessed.csv", index=False)

    print("Data berhasil disimpan!")
