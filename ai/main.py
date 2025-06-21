import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, DMatrix, train
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Loading and preprocessing the dataset
data = pd.read_csv('Maternal Health Risk Data Set.csv')
data = data.dropna()

# Encoding the target variable
label_encoder = LabelEncoder()
data['RiskLevel'] = label_encoder.fit_transform(data['RiskLevel'])

# Defining features and target
X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = data['RiskLevel']

# Compute class weights to handle imbalance (optional)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Splitting the dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Prepare DMatrix for xgboost.train with feature names as a list of strings
dtrain = DMatrix(X_train, label=y_train, feature_names=list(X.columns))
dval = DMatrix(X_val, label=y_val, feature_names=list(X.columns))
dtest = DMatrix(X_test, label=y_test, feature_names=list(X.columns))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 75, 100],
    'reg_lambda': [1, 5, 10],
    'reg_alpha': [0, 1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_val, y_train_val)  # Removed early_stopping_rounds
print("Best Parameters:", grid_search.best_params_)

# Use best parameters for training with xgboost.train with early stopping
best_params = grid_search.best_params_
evals_result = {}
best_model = train(
    params={**best_params, 'objective': 'multi:softprob', 'num_class': len(np.unique(y))},
    dtrain=dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False
)

# Use best_iteration for final predictions
best_iteration = best_model.best_iteration + 1  # +1 because best_iteration is 0-based
print(f"Best number of iterations: {best_iteration}")

# Extract evaluation results
epochs = len(evals_result['train']['mlogloss'])
x_axis = range(1, epochs + 1)

# Calculate accuracy for each iteration using a single trained model
model_iter = train(
    params={**best_params, 'objective': 'multi:softprob', 'num_class': len(np.unique(y))},
    dtrain=dtrain,
    num_boost_round=epochs,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    evals_result={},
    verbose_eval=False
)
train_accuracy = []
val_accuracy = []
for i in range(epochs):
    train_pred = model_iter.predict(dtrain, iteration_range=(0, i + 1))
    val_pred = model_iter.predict(dval, iteration_range=(0, i + 1))
    train_accuracy.append(accuracy_score(y_train, np.argmax(train_pred, axis=1)))
    val_accuracy.append(accuracy_score(y_val, np.argmax(val_pred, axis=1)))

# Predictions on test set using best_iteration
y_pred = best_model.predict(dtest, iteration_range=(0, best_iteration))
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred_labels)
print("\nTest Set Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

# Visualization 1: Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
features = X.columns
importances = best_model.get_score(importance_type='weight')
importances = [importances.get(f'f{i}', 0) for i in range(len(features))]  # Map feature indices to names
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
cm = confusion_matrix(y_test, y_pred_labels)
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

# Visualization 5: Log Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(x_axis, evals_result['train']['mlogloss'], label='Training Log Loss', color='#36A2EB')
plt.plot(x_axis, evals_result['val']['mlogloss'], label='Validation Log Loss', color='#FF6384')
plt.xlabel('Training Iteration')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('log_loss_graph.png')
plt.close()

# Cross-validation score
model_cv = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='f1_weighted')
print("\nCross-Validation f1-weighted Scores:", cv_scores)
print("Mean CV f1-weighted Score:", cv_scores.mean())
print("Standard Deviation of CV f1-weighted Score:", cv_scores.std())

# Saving the model and label encoder
joblib.dump(best_model, 'maternal_risk_model_improved.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Function to predict risk level for new data
def predict_maternal_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    input_data = DMatrix(pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    }), feature_names=list(X.columns))
    prediction = best_model.predict(input_data, iteration_range=(0, best_iteration))
    predicted_risk = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    return predicted_risk

# Example prediction
example = predict_maternal_risk(25, 130, 80, 15, 98, 86)
print("\nExample Prediction for Age=25, SystolicBP=130, DiastolicBP=80, BS=15, BodyTemp=98, HeartRate=86:")
print(f"Predicted Risk Level: {example}")

# Note: Visualizations are saved as 'feature_importance.png', 'risk_distribution.png', 'confusion_matrix.png', 'accuracy_line_graph.png', and 'log_loss_graph.png'