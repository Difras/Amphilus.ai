import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Loading and preprocessing the dataset
data = pd.read_csv('Maternal Health Risk Data Set.csv')

# Handling missing values (if any)
data = data.dropna()

# Encoding the target variable (RiskLevel: high risk, mid risk, low risk)
label_encoder = LabelEncoder()
data['RiskLevel'] = label_encoder.fit_transform(data['RiskLevel'])

# Defining features and target
X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = data['RiskLevel']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training set into training and validation sets for tracking accuracy
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initializing the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Training the model with evaluation sets to track accuracy
eval_set = [(X_train_sub, y_train_sub), (X_val, y_val)]
model.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)

# Extracting log loss from evaluation results (XGBoost doesn't track accuracy directly)
evals_result = model.evals_result()
epochs = len(evals_result['validation_0']['mlogloss'])
x_axis = range(1, epochs + 1)

# Calculating accuracy for each iteration
train_accuracy = []
val_accuracy = []
for i in range(epochs):
    # Predict on training and validation sets for each iteration
    model.set_params(n_estimators=i + 1)
    model.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)
    train_pred = model.predict(X_train_sub)
    val_pred = model.predict(X_val)
    train_accuracy.append(accuracy_score(y_train_sub, train_pred))
    val_accuracy.append(accuracy_score(y_val, val_pred))

# Retrain the model on the full training set for final predictions
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualization 1: Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
features = X.columns
importances = model.feature_importances_
plt.bar(features, importances, color=['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF', '#FF9F40'])
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance in Maternal Risk Prediction Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Visualization 2: Pie Chart for Risk Level Distribution
risk_counts = data['RiskLevel'].value_counts()
risk_labels = [label_encoder.classes_[i] for i in risk_counts.index]
plt.figure(figsize=(8, 8))
plt.pie(risk_counts, labels=risk_labels, autopct='%1.1f%%', colors=['#FF6384', '#FFCE56', '#4BC0C0'])
plt.title('Distribution of Maternal Health Risk Levels')
plt.tight_layout()
plt.savefig('risk_distribution.png')
plt.close()

# Visualization 3: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Maternal Risk Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Visualization 4: Line Graph for Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(x_axis, train_accuracy, label='Training Accuracy', color='#36A2EB', marker='o')
plt.plot(x_axis, val_accuracy, label='Validation Accuracy', color='#FF6384', marker='o')
plt.xlabel('Training Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_line_graph.png')
plt.close()

# Saving the model and label encoder
joblib.dump(model, 'maternal_risk_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Function to predict risk level for new data
def predict_maternal_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })
    prediction = model.predict(input_data)
    predicted_risk = label_encoder.inverse_transform(prediction)[0]
    return predicted_risk

# Example prediction
example = predict_maternal_risk(25, 130, 80, 15, 98, 86)
print("\nExample Prediction for Age=25, SystolicBP=130, DiastolicBP=80, BS=15, BodyTemp=98, HeartRate=86:")
print(f"Predicted Risk Level: {example}")

# Note: Visualizations are saved as 'feature_importance.png', 'risk_distribution.png', 'confusion_matrix.png', and 'accuracy_line_graph.png'