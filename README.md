# Maternal Health Risk Prediction Model

![amphilus_git](https://github.com/user-attachments/assets/d7220cb3-c253-4cbc-93bb-8d92bffb5edd)

This repository contains a Python script that builds and trains an AI model to predict maternal health risk levels (high risk, mid risk, low risk) using an XGBoost classifier. The model utilizes a dataset with features such as Age, SystolicBP, DiastolicBP, BS (Blood Sugar), BodyTemp, and HeartRate, and provides visualizations to understand the model's performance and data distribution.

## Overview

The script leverages the following libraries:

- **pandas** for data manipulation
- **scikit-learn** for preprocessing, train-test splitting, and evaluation metrics
- **xgboost** for the classification model
- **joblib** for saving the trained model and label encoder
- **matplotlib** and **seaborn** for generating visualizations

The model is trained on the provided `Maternal Health Risk Data Set.csv` and includes visualizations to analyze feature importance, risk level distribution, confusion matrix, and training/validation accuracy over iterations.

## Prerequisites

- Python 3.x
- Required libraries (install via pip):

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

## Dataset

The dataset (`Maternal Health Risk Data Set.csv`) should be placed in the same directory as the script. It contains the following columns:

- **Age**
- **SystolicBP**
- **DiastolicBP**
- **BS** (Blood Sugar)
- **BodyTemp**
- **HeartRate**
- **RiskLevel** (target variable: high risk, mid risk, low risk)

## Installation

1. Clone the repository or copy the script to your local machine
2. Install the required dependencies using the command above
3. Ensure the dataset file is in the working directory

## Usage

1. Run the script:
   ```bash
   python maternal_risk_prediction_with_line_graph.py
   ```

2. The script will:
   - Train the XGBoost model on the dataset
   - Generate predictions and evaluate the model
   - Save the trained model (`maternal_risk_model.pkl`) and label encoder (`label_encoder.pkl`)
   - Create and save visualizations as PNG files
   - Provide an example prediction for sample input data

3. **Output:**
   - Console output includes accuracy, classification report, and an example prediction
   - Visualizations are saved as `feature_importance.png`, `risk_distribution.png`, `confusion_matrix.png`, and `accuracy_line_graph.png`

## Visualizations

The script generates the following visualizations to aid in understanding the model and data:

### 1. Feature Importance Bar Chart (`feature_importance.png`)

![feature_importance](https://github.com/user-attachments/assets/cb7da24e-a2bd-4086-9368-c863803bc323)

- Displays the relative importance of each feature in the XGBoost model
- Helps identify which factors (e.g., SystolicBP, BS) most influence the risk prediction

### 2. Pie Chart for Risk Level Distribution (`risk_distribution.png`)

![risk_distribution](https://github.com/user-attachments/assets/8be57b1f-c469-45cd-a889-8cc938aacd4e)

- Shows the proportion of high risk, mid risk, and low risk cases in the dataset
- Useful for assessing class balance

### 3. Confusion Matrix Heatmap (`confusion_matrix.png`)

![confusion_matrix](https://github.com/user-attachments/assets/7fd9cd43-42e9-4198-a004-bffc597e0bb7)

- Visualizes the model's performance by showing correct and incorrect predictions
- Example illustrates the number of instances correctly classified (e.g., 41 high risk cases predicted as high risk) and misclassified (e.g., 4 low risk cases predicted as high risk)

### 4. Line Graph for Training and Validation Accuracy (`accuracy_line_graph.png`)

![accuracy_line_graph](https://github.com/user-attachments/assets/eb7726a9-9567-430d-8214-4f535b5a7455)

- Plots the training and validation accuracy over training iterations
- Indicates how the model's accuracy evolves, with training accuracy (blue) and validation accuracy (red) tracked
- Example shows training accuracy reaching ~0.95 and validation accuracy stabilizing around 0.85-0.90 after 40 iterations

## Example Prediction

The script includes a function `predict_maternal_risk` to predict risk levels for new data. An example is provided with input values:

```python
Age=25, SystolicBP=130, DiastolicBP=80, BS=15, BodyTemp=98, HeartRate=86
```

The predicted risk level is printed to the console.

## Saved Models

- **`maternal_risk_model.pkl`**: The trained XGBoost model
- **`label_encoder.pkl`**: The label encoder used to transform risk levels

These can be loaded later for predictions without retraining.

## Notes

- The line graph tracks accuracy by retraining the model at each iteration with a validation set
- The default number of boosting rounds is 100, but this can be adjusted by modifying the `n_estimators` parameter in the `XGBClassifier`
- Visualizations are saved in the working directory for easy review
- Ensure the dataset file is correctly formatted and contains no missing values (the script drops NA values by default)

## Future Improvements

- Add early stopping to optimize the number of iterations
- Include hyperparameter tuning (e.g., using GridSearchCV) to improve model performance
- Extend visualizations to include learning curves or ROC curves for more detailed analysis

## License

This project is for educational purposes. Feel free to modify and distribute, but please credit the original author.
