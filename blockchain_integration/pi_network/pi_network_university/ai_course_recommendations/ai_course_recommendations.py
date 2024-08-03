import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load course data
course_data = pd.read_csv('data/course_data.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(course_data.drop('course_id', axis=1), course_data['course_id'], test_size=0.2, random_state=42)

# Train random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Train neural network
nn = Sequential()
nn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on test set
y_pred_rfc = rfc.predict(X_test)
y_pred_nn = nn.predict(X_test)

# Evaluate model performance
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

print(f'Random Forest Classifier accuracy: {accuracy_rfc:.3f}')
print(f'Neural Network accuracy: {accuracy_nn:.3f}')

# Use models to make course recommendations
def recommend_courses(user_data):
    # Preprocess user data
    user_data = pd.DataFrame(user_data).drop('user_id', axis=1)
    
    # Make predictions on user data
    predictions_rfc = rfc.predict(user_data)
    predictions_nn = nn.predict(user_data)
    
    # Return recommended courses
    return course_data[course_data['course_id'].isin(predictions_rfc) | course_data['course_id'].isin(predictions_nn)]
