# pi_network/ai/fraud_detection_ai.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsemble
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Load the dataset
df = pd.read_csv('fraud_data.csv')

# Preprocess the data
X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_pca)
X_test_tsne = tsne.transform(X_test_pca)

# Define the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Define the model checkpoint callback
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(X_train_pca, to_categorical(y_train), epochs=100, batch_size=128, validation_data=(X_test_pca, to_categorical(y_test)), callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
y_pred = model.predict(X_test_pca)
y_pred_class = np.argmax(y_pred, axis=1)
print('Accuracy:', accuracy_score(y_test, y_pred_class))
print('Classification Report:')
print(classification_report(y_test, y_pred_class))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_class))

# Define the ensemble models
xgb_model = XGBClassifier()
cat_model = CatBoostClassifier()
lgbm_model = LGBMClassifier()
gbm_model = GradientBoostingClassifier()
voting_model = VotingClassifier(estimators=[('xgb', xgb_model), ('cat', cat_model), ('lgbm', lgbm_model), ('gbm', gbm_model)])
stacking_model = StackingClassifier(estimators=[('xgb', xgb_model), ('cat', cat_model), ('lgbm', lgbm_model), ('gbm', gbm_model)], final_estimator=LogisticRegression())
bagging_model = BaggingClassifier(base_estimator=xgb_model, n_estimators=10)
adaboost_model = AdaBoostClassifier(base_estimator=xgb_model, n_estimators=10)

# Train the ensemble models
xgb_model.fit(X_train_pca, y_train)
cat_model.fit(X_train_pca, y_train)
lgbm_model.fit(X_train_pca, y_train)
gbm_model.fit(X_train_pca, y_train)
voting_model.fit(X_train_pca, y_train)
stacking_model.fit(X_train_pca, y_train)
bagging_model.fit(X_train_pca, y_train)
adaboost_model.fit(X_train_pca, y_train)

# Evaluate the ensemble models
y_pred_xgb = xgb_model.predict(X_test_pca)
y_pred_cat = cat_model.predict(X_test_pca)
y_pred_lgbm = lgbm_model.predict(X_test_pca)
y_pred_gbm = gbm_model.predict(X_test_pca)
y_pred_voting = voting_model.predict(X_test_pca)
y_pred_stacking = stacking_model.predict(X_test_pca)
y_pred_bagging = bagging_model.predict(X_test_pca)
y_pred_adaboost = adaboost_model.predict(X_test_pca)

print('XGB Accuracy:', accuracy_score(y_test, y_pred_xgb))
print('CAT Accuracy:', accuracy_score(y_test, y_pred_cat))
print('LGBM Accuracy:', accuracy_score(y_test, y_pred_lgbm))
print('GBM Accuracy:', accuracy_score(y_test, y_pred_gbm))
print('Voting Accuracy:', accuracy_score(y_test, y_pred_voting))
print('Stacking Accuracy:', accuracy_score(y_test, y_pred_stacking))
print('Bagging Accuracy:', accuracy_score(y_test, y_pred_bagging))
print('AdaBoost Accuracy:', accuracy_score(y_test, y_pred_adaboost))

# Define the feature selection models
rf_model = RandomForestClassifier()
svm_model = SVC(kernel='linear', probability=True)
gnb_model = GaussianNB()
qda_model = QuadraticDiscriminantAnalysis()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression()

# Train the feature selection models
rf_model.fit(X_train_pca, y_train)
svm_model.fit(X_train_pca, y_train)
gnb_model.fit(X_train_pca, y_train)
qda_model.fit(X_train_pca, y_train)
dt_model.fit(X_train_pca, y_train)
knn_model.fit(X_train_pca, y_train)
lr_model.fit(X_train_pca, y_train)

# Evaluate the feature selection models
y_pred_rf = rf_model.predict(X_test_pca)
y_pred_svm = svm_model.predict(X_test_pca)
y_pred_gnb = gnb_model.predict(X_test_pca)
y_pred_qda = qda_model.predict(X_test_pca)
y_pred_dt = dt_model.predict(X_test_pca)
y_pred_knn = knn_model.predict(X_test_pca)
y_pred_lr = lr_model.predict(X_test_pca)

print('RF Accuracy:', accuracy_score(y_test, y_pred_rf))
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print('GNB Accuracy:', accuracy_score(y_test, y_pred_gnb))
print('QDA Accuracy:', accuracy_score(y_test, y_pred_qda))
print('DT Accuracy:', accuracy_score(y_test, y_pred_dt))
print('KNN Accuracy:', accuracy_score(y_test, y_pred_knn))
print('LR Accuracy:', accuracy_score(y_test, y_pred_lr))

# Define the hyperparameter tuning models
grid_search = GridSearchCV(xgb_model, {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.5, 1.0]}, cv=5, scoring='accuracy')
random_search = RandomizedSearchCV(xgb_model, {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.5, 1.0]}, cv=5, scoring='accuracy', n_iter=10)

# Train the hyperparameter tuning models
grid_search.fit(X_train_pca, y_train)
random_search.fit(X_train_pca, y_train)

# Evaluate the hyperparameter tuning models
print('Grid Search Best Parameters:', grid_search.best_params_)
print('Grid Search Best Score:', grid_search.best_score_)
print('Random Search Best Parameters:', random_search.best_params_)
print('Random Search Best Score:', random_search.best_score_)

# Define the feature importance models
feature_importance = rf_model.feature_importances_
print('Feature Importance:')
print(feature_importance)

# Define the partial dependence plots
partial_dependence = pd.DataFrame({'Feature 1': X_test_pca[:, 0], 'Feature 2': X_test_pca[:, 1], 'Target': y_test})
sns.lmplot(x='Feature 1', y='Target', data=partial_dependence, hue='is_fraud')
sns.lmplot(x='Feature 2', y='Target', data=partial_dependence, hue='is_fraud')

# Define the learning curves
train_sizes, train_scores, test_scores = learning_curve(xgb_model, X_train_pca, y_train, cv=5, scoring='accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training Score')
plt.plot(train_sizes, test_scores_mean, label='Test Score')
plt.legend()
plt.show()

# Define the ROC-AUC curve
y_pred_proba = model.predict(X_test_pca)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print('ROC-AUC Score:', roc_auc)

# Define the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
print('Average Precision Score:', average_precision)

# Define the F1 score
f1 = f1_score(y_test, y_pred_class)
print('F1 Score:', f1)

# Define the recall score
recall = recall_score(y_test, y_pred_class)
print('Recall Score:', recall)

# Define the precision score
precision = precision_score(y_test, y_pred_class)
print('Precision Score:', precision)

# Define the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
print('Confusion Matrix:')
print(cm)

# Define the classification report
cr = classification_report(y_test, y_pred_class)
print('Classification Report:')
print(cr)

# Define the feature selection pipeline
feature_selection_pipeline = Pipeline([
    ('feature_selection', SelectFromModel(rf_model)),
    ('classifier', xgb_model)
])

# Train the feature selection pipeline
feature_selection_pipeline.fit(X_train_pca, y_train)

# Evaluate the feature selection pipeline
y_pred_fs = feature_selection_pipeline.predict(X_test_pca)
print('Feature Selection Accuracy:', accuracy_score(y_test, y_pred_fs))

# Define the recursive feature elimination pipeline
rfe_pipeline = Pipeline([
    ('feature_selection', RFECV(rf_model, cv=5, scoring='accuracy')),
    ('classifier', xgb_model)
])

# Train the recursive feature elimination pipeline
rfe_pipeline.fit(X_train_pca, y_train)

# Evaluate the recursive feature elimination pipeline
y_pred_rfe = rfe_pipeline.predict(X_test_pca)
print('Recursive Feature Elimination Accuracy:', accuracy_score(y_test, y_pred_rfe))

# Define the hyperparameter tuning pipeline
hyperparameter_tuning_pipeline = Pipeline([
    ('feature_selection', SelectFromModel(rf_model)),
    ('classifier', GridSearchCV(xgb_model, {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.5, 1.0]}, cv=5, scoring='accuracy'))
])

# Train the hyperparameter tuning pipeline
hyperparameter_tuning_pipeline.fit(X_train_pca, y_train)

# Evaluate the hyperparameter tuning pipeline
y_pred_ht = hyperparameter_tuning_pipeline.predict(X_test_pca)
print('Hyperparameter Tuning Accuracy:', accuracy_score(y_test, y_pred_ht))
