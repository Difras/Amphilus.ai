import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    filename="maternal_risk_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MaternalRiskModel:
    def __init__(self, model_path="maternal_risk_model.pkl", encoder_path="label_encoder.pkl", scaler_path="scaler.pkl"):
        """Initialize model with paths for saving/loading artifacts."""
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

    def preprocess_data(self, data_path):
        """Load and preprocess dataset."""
        try:
            data = pd.read_csv(data_path)
            logger.info("Loaded dataset with %d rows", len(data))

            # Handle missing values
            data = data.dropna()
            if data.empty:
                raise ValueError("Dataset is empty after dropping missing values.")

            # Outlier detection and capping
            for col in self.feature_names:
                q1 = data[col].quantile(0.01)
                q99 = data[col].quantile(0.99)
                data[col] = data[col].clip(q1, q99)

            # Encode target variable
            self.label_encoder = LabelEncoder()
            data['RiskLevel'] = self.label_encoder.fit_transform(data['RiskLevel'])
            logger.info("Encoded target variable: %s", self.label_encoder.classes_)

            # Feature engineering
            data['BP_Difference'] = data['SystolicBP'] - data['DiastolicBP']
            data['Age_BS_Interaction'] = data['Age'] * data['BS']
            self.feature_names.extend(['BP_Difference', 'Age_BS_Interaction'])

            # Split features and target
            X = data[self.feature_names]
            y = data['RiskLevel']

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, y, data
        except Exception as e:
            logger.error("Preprocessing failed: %s", str(e))
            raise

    def train_model(self, X, y):
        """Train XGBoost model with hyperparameter tuning and cross-validation."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Define model
            base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logger.info("Best parameters: %s", grid_search.best_params_)
            logger.info("Best CV accuracy: %.4f", grid_search.best_score_)

            # Train final model
            eval_set = [(X_train_sub, y_train_sub), (X_val, y_val)]
            self.model.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            logger.info("Cross-validation scores: %s (mean: %.4f)", cv_scores, cv_scores.mean())

            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info("Test set accuracy: %.4f", accuracy)
            print("Test Set Accuracy:", accuracy)
            print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

            return X_train_sub, y_train_sub, X_val, y_val, X_test, y_test
        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise

    def visualize_results(self, X_train, y_train, X_val, y_val, X_test, y_test, data):
        """Generate visualizations."""
        try:
            # Feature Importance (Interactive Plotly Bar)
            fig = px.bar(
                x=self.model.feature_importances_,
                y=self.feature_names,
                title='Feature Importance in Maternal Risk Prediction',
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=self.feature_names,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(xaxis_title="Importance Score", yaxis_title="Features")
            fig.write_html('feature_importance.html')

            # Risk Distribution (Interactive Plotly Pie)
            risk_counts = data['RiskLevel'].value_counts()
            risk_labels = [self.label_encoder.classes_[i] for i in risk_counts.index]
            fig = px.pie(
                values=risk_counts,
                names=risk_labels,
                title='Distribution of Maternal Health Risk Levels',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.write_html('risk_distribution.html')

            # Confusion Matrix (Seaborn Heatmap)
            cm = confusion_matrix(y_test, self.model.predict(X_test))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.close()

            # Training and Validation Accuracy (Interactive Plotly Line)
            evals_result = self.model.evals_result()
            epochs = len(evals_result['validation_0']['mlogloss'])
            x_axis = list(range(1, epochs + 1))
            train_accuracy = []
            val_accuracy = []
            for i in range(epochs):
                temp_model = XGBClassifier(**self.model.get_params(), n_estimators=i + 1)
                temp_model.fit(X_train, y_train, verbose=False)
                train_accuracy.append(accuracy_score(y_train, temp_model.predict(X_train)))
                val_accuracy.append(accuracy_score(y_val, temp_model.predict(X_val)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=train_accuracy, mode='lines+markers', name='Training Accuracy', line=dict(color='#36A2EB')))
            fig.add_trace(go.Scatter(x=x_axis, y=val_accuracy, mode='lines+markers', name='Validation Accuracy', line=dict(color='#FF6384')))
            fig.update_layout(
                title='Training and Validation Accuracy Over Iterations',
                xaxis_title='Training Iteration',
                yaxis_title='Accuracy',
                showlegend=True,
                grid=dict(xaxis=dict(visible=True), yaxis=dict(visible=True))
            )
            fig.write_html('accuracy_line_graph.html')

            # SHAP Summary Plot
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            plt.close()
            logger.info("Generated visualizations: feature_importance.html, risk_distribution.html, confusion_matrix.png, accuracy_line_graph.html, shap_summary.png")
        except Exception as e:
            logger.error("Visualization failed: %s", str(e))
            raise

    def save_artifacts(self):
        """Save model, encoder, and scaler."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Saved artifacts: %s, %s, %s", self.model_path, self.encoder_path, self.scaler_path)
        except Exception as e:
            logger.error("Saving artifacts failed: %s", str(e))
            raise

    def predict_risk(self, age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
        """Predict risk level for new data with input validation."""
        try:
            # Input validation
            inputs = {
                'Age': (age, 10, 100),
                'SystolicBP': (systolic_bp, 50, 200),
                'DiastolicBP': (diastolic_bp, 30, 150),
                'BS': (bs, 0, 50),
                'BodyTemp': (body_temp, 90, 110),
                'HeartRate': (heart_rate, 40, 120)
            }
            for name, (value, min_val, max_val) in inputs.items():
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    raise ValueError(f"{name} must be a number between {min_val} and {max_val}")

            # Create input DataFrame
            input_data = pd.DataFrame({
                'Age': [age],
                'SystolicBP': [systolic_bp],
                'DiastolicBP': [diastolic_bp],
                'BS': [bs],
                'BodyTemp': [body_temp],
                'HeartRate': [heart_rate],
                'BP_Difference': [systolic_bp - diastolic_bp],
                'Age_BS_Interaction': [age * bs]
            })

            # Scale input
            input_scaled = self.scaler.transform(input_data)

            # Predict
            prediction = self.model.predict(input_scaled)
            predicted_risk = self.label_encoder.inverse_transform(prediction)[0]
            logger.info("Prediction for input %s: %s", input_data.to_dict(), predicted_risk)
            return predicted_risk
        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            raise

def main():
    """Main function to run the model pipeline."""
    try:
        model = MaternalRiskModel()
        X, y, data = model.preprocess_data('Maternal Health Risk Data Set.csv')
        X_train, y_train, X_val, y_val, X_test, y_test = model.train_model(X, y)
        model.visualize_results(X_train, y_train, X_val, y_val, X_test, y_test, data)
        model.save_artifacts()

        # Example prediction
        example = model.predict_risk(25, 130, 80, 15, 98, 86)
        print("\nExample Prediction:")
        print(f"Predicted Risk Level: {example}")
    except Exception as e:
        logger.error("Main pipeline failed: %s", str(e))
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()