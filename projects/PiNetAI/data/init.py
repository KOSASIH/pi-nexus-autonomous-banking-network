import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
node_data = pd.read_csv('data/node_data.csv')
transaction_data = pd.read_csv('data/transaction_data.csv')

# Merge data
data = pd.merge(node_data, transaction_data, on='node_id')

# Preprocess data
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['amount'] = data['amount'].astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('category', axis=1), data['category'], test_size=0.2, random_state=42)

# Train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Visualize results
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.countplot(x='category', data=data)
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Save model
rf.save('model.pkl')
