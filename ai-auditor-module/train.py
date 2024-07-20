# ai-auditor-module/train.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv("smart_contract_data.csv")

# Preprocess data
X = dataset.drop(["label"], axis=1)
y = dataset["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")

# Save the model

joblib.dump(model, "ai_model.joblib")
