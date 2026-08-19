import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------------------------
# Load dataset
# ------------------------------
data = pd.read_csv("data/incubator_qc_dataset.csv")

# Features (include airflow)
X = data[['temp_error', 'humidity_error', 'oxygen_error', 'noise_level',
          'airflow', 'device_age', 'last_maintenance_days', 'repair_history']]
y = data['qc_status']

# ------------------------------
# Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Create and train Random Forest
# ------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------
# Evaluate accuracy
# ------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# ------------------------------
# Save model
# ------------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "incubator_qc_model.pkl")
joblib.dump(model, model_path)
print(f"QC model trained and saved successfully at {model_path}!")