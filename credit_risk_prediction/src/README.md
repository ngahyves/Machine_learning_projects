## Introduction

**How can we detect the 0.17% of fraudulent transactions in a data stream without negatively impacting millions of legitimate customers?** This project addresses this challenge by developing an end-to-end machine learning solution.

Starting with a highly imbalanced public dataset, this project demonstrates a rigorous methodology for building and optimising a high-performance model. The comparative ‘baseline vs. optimised’ approach has enabled **an 89% increase in fraud detection**.

The final solution is an **XGBoost model (AUC of 0.93)**, deployed via a **REST API (FastAPI)**, thus demonstrating a complete data lifecycle, from analysis to production.
## Methodology

The project follows a complete 6-step life cycle:

1.  **Exploratory Data Analysis (EDA):** Identification of extreme class imbalance (0.17% fraud) and analysis of variable distributions to understand the data.
2.  **Data Pre-processing:** Scaling of the `Amount` and `Time` variables with `StandardScaler` to normalise their distributions.
3.  **Imbalance Management:** Application of the **SMOTE** oversampling technique on the training set to enable the model to learn effectively from the minority class.
4.  **Comparative Modelling:**
*   Training **baseline models** (Logistic Regression, RandomForest, XGBoost) on the raw data.
*   Training **optimised models** on the prepared data (scaling + SMOTE) to measure the impact of data preparation.
5.  **Rigorous Evaluation:** Use metrics adapted to imbalanced data (Confusion Matrix, Classification Ratio, ROC-AUC curve) to select the best model.
6.    **Deployment:** Exposure of the final XGBoost model via a REST API with FastAPI to enable real-time predictions.

## Key Results

### 1. Impact of Pre-processing

The comparison between the reference model and the optimised model showed a dramatic improvement in performance.
| Metric          | Baseline XGBoost | Optimized XGBoost (SMOTE) | Relative Impact |
| :-------------- | :--------------: | :-----------------------: | :-------------: |
| **AUC**         |       0.82       |          **0.93**         |     **+13%**    |
| **Recall (Fraud)**  |       0.35       |          **0.66**         |     **+89%**    |
| **F1-Score (Fraud)**|       0.45       |          **0.79**         |     **+76%**    |

### 2. Performance of the Final Model

The optimised XGBoost model was selected for deployment.
![image](https://github.com/user-attachments/assets/e843bbff-be8e-40c4-96f4-18d13614d42a)

### 3. Key Fraud Indicators (Feature Importance)

A comparative analysis of the three models revealed that variables **V14, V10, V12, V4, and V17** were the most robust and reliable predictors.
![image](https://github.com/user-attachments/assets/8dcb1274-83b8-4a5d-a40a-ece1a9116077)

## How to Use this Project

### 1. Prerequisites

Make sure you have Python 3.8+ installed.

### 2. Installation

Clone this repository and install the dependencies:

```bash
git clone [https://github.com/ngahyves/Machine_learning_projects/tree/main]
cd [Machine_learning_projects]
pip install -r requirements.txt

