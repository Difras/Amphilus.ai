import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.impute import SimpleImputer

# Loading and preprocessing
maternal_data = pd.read_csv('Maternal Health Risk Data Set.csv')
maternal_data = maternal_data.dropna()
maternal_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
maternal_data = maternal_data[maternal_features + ['RiskLevel']]

fetal_data = pd.read_csv('fetal_health.csv')
fetal_data = fetal_data.dropna()
fetal_features = [
    'baseline value', 'accelerations', 'uterine_contractions',
    'light_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability'
]
fetal_data['RiskLevel'] = fetal_data['fetal_health'].map({1: 'low risk', 2: 'mid risk', 3: 'high risk'})
fetal_data = fetal_data[fetal_features + ['RiskLevel']]

fetal_data.columns = [
    'FetalHeartRate', 'Accelerations', 'UterineContractions',
    'LightDecelerations', 'ProlonguedDecelerations', 'AbnormalShortTermVariability', 'RiskLevel'
]

combined_data = pd.concat([maternal_data, fetal_data], axis=0, ignore_index=True)

# Define features list before imputation
features = [
    'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
    'FetalHeartRate', 'Accelerations', 'UterineContractions',
    'LightDecelerations', 'ProlonguedDecelerations', 'AbnormalShortTermVariability'
]

# Dynamic outlier handling using IQR
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

combined_data = cap_outliers(combined_data, 'FetalHeartRate')

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
combined_data[features] = imputer.fit_transform(combined_data[features])

label_encoder = LabelEncoder()
combined_data['RiskLevel'] = label_encoder.fit_transform(combined_data['RiskLevel'])

X = combined_data[features]
y = combined_data['RiskLevel']

# Robust scaling to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Grid Search for hyperparameter tuning
param_grid = {
    'max_depth': [6, 8],                # Fewer options to reduce memory usage
    'learning_rate': [0.05],            # Single value for speed
    'n_estimators': [200, 300],         # Fewer options
    'scale_pos_weight': [1.0]           # Single value
}
# Enable GPU acceleration by setting tree_method and predictor
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',     # CPU is faster for small data
    predictor='auto'
)
model.fit(X_train, y_train)
best_model = model

# Evaluate with validation set
eval_set = [(X_train, y_train), (X_val, y_val)]
best_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

results = best_model.evals_result()
iterations = range(len(results['validation_0']['mlogloss']))
train_accuracy = [1 - loss for loss in results['validation_0']['mlogloss']]
val_accuracy = [1 - loss for loss in results['validation_1']['mlogloss']]

# Final training and prediction
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualizations
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, importance_type='weight')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_combined.png')
plt.close()

plt.figure(figsize=(8, 6))
combined_data['RiskLevel'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=label_encoder.classes_)
plt.title('Risk Level Distribution')
plt.ylabel('')
plt.savefig('risk_distribution_combined.png')
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_combined.png')
plt.close()

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
    pickle.dump(best_model, f)
with open('label_encoder_combined.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('scaler_combined.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Prediction function
def predict_combined_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,
                         fetal_heart_rate, accelerations, uterine_contractions,
                         light_decelerations, prolongued_decelerations, abnormal_short_term_variability):
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,
                            min(fetal_heart_rate, 180), accelerations, uterine_contractions,
                            light_decelerations, prolongued_decelerations, abnormal_short_term_variability]])
    input_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_scaled)
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

# Additional test cases for validation
test_cases = [
    {
        'age': 35, 'systolic_bp': 160, 'diastolic_bp': 100, 'bs': 9, 'body_temp': 99, 'heart_rate': 90,
        'fetal_heart_rate': 180, 'accelerations': 0.002, 'uterine_contractions': 0.008,
        'light_decelerations': 0.01, 'prolongued_decelerations': 0.01, 'abnormal_short_term_variability': 50
    },  # Expected high risk
    {
        'age': 25, 'systolic_bp': 120, 'diastolic_bp': 80, 'bs': 6.0, 'body_temp': 98, 'heart_rate': 70,
        'fetal_heart_rate': 130, 'accelerations': 0.01, 'uterine_contractions': 0.004,
        'light_decelerations': 0.0, 'prolongued_decelerations': 0.0, 'abnormal_short_term_variability': 10
    }  # Expected low risk
]
for i, tc in enumerate(test_cases):
    pred = predict_combined_risk(**tc)
    print(f"\nPredicted Risk Level for test case {i+1}: {pred}")