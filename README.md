# DDoS Attack Detection using Machine Learning

## Overview:
This project focuses on detecting Distributed Denial of Service (DDoS) attacks using machine learning models. A real-world dataset is used, and three machine learning algorithms were applied: Random Forest, Logistic Regression, and Neural Network (MLPClassifier). The workflow covers the essential steps from data preprocessing to model evaluation and comparison.

## Libraries Used:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms, model training, evaluation, and metrics..

![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/your-repo-name?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/your-repo-name)
![GitHub issues](https://img.shields.io/github/issues/yourusername/your-repo-name)
![GitHub forks](https://img.shields.io/github/forks/yourusername/your-repo-name?style=social)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange)
![Data Source](https://img.shields.io/badge/dataset-Kaggle-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

## Workflow Steps:
1. **Loading Libraries**: Importing the necessary libraries for data analysis, machine learning, and evaluation.
2. **Data Loading**: Loading the dataset that contains both normal and attack traffic data.
3. **Data Preprocessing**:
   - Converting categorical data into dummy/indicator variables (if applicable).
   - Normalizing or standardizing the data using `StandardScaler` to ensure consistent scaling for certain models.
4. **Exploratory Data Analysis (EDA)**: 
   - Analyzing the distribution of features using visualization techniques (e.g., distribution plots).
5. **Data Splitting**: 
   - Dividing the dataset into training and testing sets using the `train_test_split()` method.
6. **Model Training**: 
   - Training three different models:
     1. **Random Forest**: An ensemble model using decision trees.
     2. **Logistic Regression**: A linear model for binary classification.
     3. **Neural Network (MLPClassifier)**: A model using a multi-layer perceptron.
7. **Model Evaluation**: 
   - Using metrics such as accuracy, F1 score, precision, recall, and confusion matrices to evaluate model performance.
   - Plotting ROC curves to assess the models' classification effectiveness.
8. **Model Comparison**: 
   - Comparing the performance of each model based on evaluation metrics.

## Model Performance:
1. **Random Forest**:
   - **Accuracy**: 0.9995
   - **F1 Score**: 0.9995
   - **Precision**: 1.0000
   - **Recall**: 0.9990
   
   **Observation**: The Random Forest classifier achieved nearly perfect accuracy and F1 score, indicating excellent performance in detecting DDoS attacks.

2. **Logistic Regression**:
   - **Accuracy**: 0.9447
   - **F1 Score**: 0.9498
   - **Precision**: 0.9100
   - **Recall**: 0.9933

   **Observation**: Logistic Regression performed well but had slightly lower accuracy and precision compared to Random Forest. However, it maintained a high recall, meaning it effectively identified most attack cases.

3. **Neural Network (MLPClassifier)**:
   - **Accuracy**: 0.9841
   - **F1 Score**: 0.9850
   - **Precision**: 0.9802
   - **Recall**: 0.9898

   **Observation**: The Neural Network also performed well, balancing between high accuracy and strong F1 score. Its performance falls between Random Forest and Logistic Regression in terms of precision and recall.

## Conclusion:
This project demonstrates the effectiveness of three machine learning models in detecting DDoS attacks. The Random Forest model shows the best performance, closely followed by the Neural Network, while Logistic Regression performs slightly lower. The comparison highlights the trade-offs in precision, recall, and overall performance.

## Final Result 
![Screenshot 2024-10-21 114624](https://github.com/user-attachments/assets/5ff8bee1-b484-49f0-beec-cf9f6be5ce7c)


