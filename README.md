# ML Assignment 2 - Streamlit Classification App

## Problem statement

This project implements the required machine learning classification workflow for Assignment 2. The goal is to train multiple classifiers on one qualifying public dataset, compare their performance using the required evaluation metrics, and present the results through an interactive Streamlit web application.

## Dataset description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Task type: Binary classification
- Instances: 569
- Features: 30 numerical features
- Target: `diagnosis`
- Class labels: `0 = malignant`, `1 = benign`

This dataset satisfies the assignment constraints of at least 12 features and at least 500 instances.

## GitHub Repository Link

- GitHub Repository: [https://github.com/RangariPrafulla/ML_Assign2](https://github.com/RangariPrafulla/ML_Assign2)
- Streamlit App: [https://mlassign2.streamlit.app/](https://mlassign2.streamlit.app/)

## Models used

The following models are trained and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

The evaluation metrics required in the assignment are:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9142 | 0.9429 | 0.9167 | 0.9296 | 0.8139 |
| K-Nearest Neighbors | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest | 0.9474 | 0.9931 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Model observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best overall model on the held-out test set. It achieved the highest accuracy, strongest MCC, and one of the best AUC values with only two total misclassifications. |
| Decision Tree | Most interpretable model, but it showed the weakest overall generalization. It produced the lowest accuracy, AUC, and MCC among the five models. |
| K-Nearest Neighbors | Very strong performer after scaling the features. It reached perfect recall for the benign class and ranked second overall by accuracy. |
| Gaussian Naive Bayes | Delivered solid and balanced results with a strong AUC, but its accuracy and MCC stayed below the top linear and distance-based models. |
| Random Forest | Achieved a high AUC and balanced precision-recall performance, but it still trailed Logistic Regression and K-Nearest Neighbors on final accuracy. |
| Overall Winner for your dataset? | Logistic Regression is the winner for this dataset because it achieved the best combination of accuracy, AUC, F1, and MCC on the test split. |

## Project structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- artifacts/
|   |-- metrics.json
|   |-- model_comparison.csv
|-- model/
|   |-- train_models.py
|   |-- *.joblib
```

## How to run locally

1. Create a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Run the training script:

```bash
python model/train_models.py
```

4. Start the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit app features

- CSV upload support for test data
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix
- Classification report
- Preview of model predictions

## Files included for submission

- Source code for training and the app
- `requirements.txt`
- `README.md`
- `test_data.csv`
- Saved model files in the `model/` folder
