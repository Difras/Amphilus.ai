import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Loading and preprocessing maternal health dataset
maternal_data = pd.read_csv('Maternal Health Risk Data Set.csv')
maternal_data = maternal_data.dropna()  # Remove any missing values (none expected)
maternal_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
maternal_data = maternal_data[maternal_features + ['RiskLevel']]

# Loading and preprocessing fetal health dataset
fetal_data = pd.read_csv('fetal_health.csv')
fetal_data = fetal_data.dropna()  # Remove any missing values (none expected)
# Select key fetal features to avoid high dimensionality
fetal_features = [
    'baseline value', 'accelerations', 'uterine_contractions',
    'light_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability'
]
# Map fetal_health to RiskLevel categories
fetal_data['RiskLevel'] = fetal_data['fetal_health'].map({1: 'low risk', 2: 'mid risk', 3: 'high risk'})
fetal_data = fetal_data[fetal_features + ['RiskLevel']]

# Renaming fetal features to avoid conflicts and ensure clarity
fetal_data.columns = [
    'FetalHeartRate', 'Accelerations', 'UterineContractions',
    'LightDecelerations', 'ProlonguedDecelerations', 'AbnormalShortTermVariability', 'RiskLevel'
]

# Combining datasets
combined_data = pd.concat([maternal_data, fetal_data], axis=0, ignore_index=True)
combined_data = combined_data.fillna(0)  # Fill missing features with 0 (e.g., fetal features for maternal rows)

# Encoding the target variable
label_encoder = LabelEncoder()
combined_data['RiskLevel'] = label_encoder.fit_transform(combined_data['RiskLevel'])

# Defining features and target
features = [
    'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
    'FetalHeartRate', 'Accelerations', 'UterineContractions',
    'LightDecelerations', 'ProlonguedDecelerations', 'AbnormalShortTermVariability'
]
X = combined_data[features]
y = combined_data['RiskLevel']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Further splitting training data into sub-training and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Initializing and training the XGBoost model with evaluation
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=100)
eval_set = [(X_subtrain, y_subtrain), (X_val, y_val)]
model.fit(X_subtrain, y_subtrain, eval_set=eval_set, verbose=True)

# Extracting training and validation accuracy from evaluation results
results = model.evals_result()
iterations = range(len(results['validation_0']['mlogloss']))
train_accuracy = [1 - loss for loss in results['validation_0']['mlogloss']]  # Approximate accuracy from logloss
val_accuracy = [1 - loss for loss in results['validation_1']['mlogloss']]   # Approximate accuracy from logloss

# Training final model on full training data
model.fit(X_train, y_train)

# Evaluating on test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualizing feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_combined.png')
plt.close()

# Visualizing risk distribution
plt.figure(figsize=(8, 6))
combined_data['RiskLevel'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=label_encoder.classes_)
plt.title('Risk Level Distribution')
plt.ylabel('')
plt.savefig('risk_distribution_combined.png')
plt.close()

# Visualizing confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_combined.png')
plt.close()

# Visualizing training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_accuracy, label='Training Accuracy')
plt.plot(iterations, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_line_graph_combined.png')
plt.close()

# Saving the model and encoders
with open('combined_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder_combined.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('scaler_combined.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Defining prediction function
def predict_combined_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,
                         fetal_heart_rate, accelerations, uterine_contractions,
                         light_decelerations, prolongued_decelerations, abnormal_short_term_variability):
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,
                            fetal_heart_rate, accelerations, uterine_contractions,
                            light_decelerations, prolongued_decelerations, abnormal_short_term_variability]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return label_encoder.inverse_transform(prediction)[0]

# Example prediction
example_input = {
    'age': 23,
    'systolic_bp': 140,
    'diastolic_bp': 80,
    'bs': 7.01,
    'body_temp': 98,
    'heart_rate': 70,
    'fetal_heart_rate': 120,
    'accelerations': 0.005,
    'uterine_contractions': 0.006,
    'light_decelerations': 0.0,
    'prolongued_decelerations': 0.0,
    'abnormal_short_term_variability': 20
}
predicted_risk = predict_combined_risk(**example_input)
print(f"\nPredicted Risk Level for example input: {predicted_risk}")