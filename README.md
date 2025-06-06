# DDoS Attack Detection Using Machine Learning

## Overview  
This project focuses on detecting **Distributed Denial of Service (DDoS)** attacks using supervised machine learning techniques applied to network traffic data. The dataset includes network flow-level features such as IP addresses, ports, protocols, packet size, and frequency metrics. The objective is to classify traffic into **benign or DDoS attack** categories based on these characteristics.

We implemented and evaluated four models:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

Performance was measured using **accuracy**, and the best models achieved over 99% test accuracy. Preprocessing and feature engineering steps were crucial in enhancing performance.

---

## Files in the Repository

### 1. `DDoS_Detection.ipynb`
Contains full implementation including:

#### Dataset Preprocessing:
- Dropping irrelevant fields (e.g., raw IPs, timestamps if needed)
- Label encoding of categorical fields (e.g., protocols)
- Feature scaling using StandardScaler
- Train/test split with stratification

#### Feature Engineering:
- Protocol layer categorization (e.g., Transport, Application layer)
- Packet-level metrics (length, rate)
- Encoded port information
- Custom feature mappings if applicable

#### Model Training:
- Logistic Regression with `liblinear` solver
- Support Vector Machine with RBF kernel
- Random Forest Classifier with tuned estimators
- XGBoost with early stopping

#### Evaluation & Visualization:
- Accuracy comparison
- Confusion matrix
- Feature importance (for RF and XGBoost)

---

## Project Workflow

### Step 1: Preprocessing
- Drop unused fields
- Encode categorical variables (IP, Port, Protocol)
- Standardize numerical features
- Train-test split with 80/20 stratified sampling

### Step 2: Model Training
Train the following models for binary classification:
- **Logistic Regression** – linear baseline
- **Support Vector Machine (SVM)** – margin-based
- **Random Forest** – ensemble of decision trees
- **XGBoost** – gradient-boosted trees

### Step 3: Model Evaluation
- Evaluate using test set accuracy
- Confusion matrix analysis for TP/FP/FN rates
- Feature importance for explainability

---

## Results Summary

| Model               | Accuracy   |
|--------------------|------------|
| XGBoost            | **99.81%** |
| Random Forest      | **99.81%** |
| Logistic Regression| 99.47%     |
| SVM                | 99.46%     |

> XGBoost and Random Forest delivered the highest accuracy on the test set, suggesting strong performance on this dataset.

---

## Key Insights

- **XGBoost** and **Random Forest** demonstrated exceptional performance and robustness.
- **Logistic Regression** and **SVM** also performed very well with minimal overfitting, serving as reliable baselines.
- Features related to **packet frequency**, **destination port**, and **transport layer** were crucial for classification.

---

## Future Work

- Implement real-time DDoS detection using streaming tools like Kafka + Apache Spark.
- Test the models on imbalanced or more diverse datasets.
- Experiment with deep learning methods (e.g., LSTM for sequential packet analysis).
- Conduct feature selection and dimensionality reduction to streamline performance.
