import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data from database or API
data = pd.read_sql("SELECT * FROM transactions", db_connection)

# Preprocess data
data.dropna(inplace=True)
data["timestamp"] = pd.to_datetime(data["timestamp"])

# Visualize data
plt.figure(figsize=(12, 6))
plt.plot(data["timestamp"], data["value"])
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Transaction Value Over Time")
plt.show()

# Perform machine learning tasks
X = data[["sender_balance", "receiver_balance"]]
y = data["value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
