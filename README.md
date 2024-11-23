# Breast-Cancer-Data-Analysis-and-Streamlit-App

## Report on Breast Cancer Prediction Using Artificial Neural Networks with Streamlit App

Submitted By:
Aishwarya Karki
Student ID: c0903073

Submitted to : Dr. Ishant Gupta 

## Abstract

Breast cancer is a significant health concern that requires early and accurate diagnosis to improve treatment outcomes. Machine learning provides a robust method for automating diagnosis based on clinical data. This project develops a predictive model using Artificial Neural Networks (ANN) to classify breast tumors as malignant or benign. The project also incorporates an interactive web application built with Streamlit, allowing users to input clinical measurements and receive predictions. The model achieved an accuracy of 94%, demonstrating its potential as a reliable diagnostic tool.

# 1. Introduction

## 1.1 Problem Statement
Breast cancer is one of the most common cancers worldwide. Early detection of malignant tumors is critical for effective treatment and better survival rates. Traditional diagnostic methods, while accurate, often rely on manual analysis, which can be time-consuming and prone to human error. Machine learning can automate these processes, delivering consistent and accurate predictions rapidly.

## 1.2 Objectives
This project aims to:
Develop a machine learning model for predicting whether a breast tumor is malignant or benign.
Enhance the model’s efficiency and interpretability through feature selection.
Provide an intuitive web application to make the predictions accessible to users.

## 1.3 Dataset Overview
The Breast Cancer dataset from Scikit-learn contains:
Features: 30 numerical attributes describing tumor characteristics (e.g., mean radius, texture, perimeter).
Target Variable: A binary classification (0: Benign, 1: Malignant).
Samples: 569 data points.

# 2. Methodology

## 2.1 Data Preprocessing
Normalization:
Features were standardized using StandardScaler to ensure uniform scaling, improving model performance.
Handling Missing Values:
The dataset had no missing values, so no imputation was necessary.

## 2.2 Feature Selection
Technique Used:
The SelectKBest method, based on ANOVA F-statistics, was applied to select the 10 most relevant features.
Selected Features:
mean radius, mean perimeter, mean area, mean concavity, mean concave points, worst radius, worst perimeter, worst area, worst concavity, worst concave points

## 2.3 Model Development
Model:
An Artificial Neural Network (ANN) was implemented using Scikit-learn’s MLPClassifier.
Architecture:
Two hidden layers with 50 neurons each.
ReLU activation function for non-linear transformations.
Adam optimizer for adaptive learning rates.
Hyperparameter Tuning:
GridSearchCV was used to optimize hidden layer sizes, activation functions, and learning rates.
Training:
The model was trained on the selected features, splitting the dataset into training and testing subsets.

## 2.4 Evaluation Metrics
The model was evaluated using:
Accuracy: Overall classification correctness.
Confusion Matrix: Breakdown of correct and incorrect predictions.
Classification Report: Precision, recall, and F1-score for both classes.

## 2.5 Interactive Application Development
Framework Used:
Streamlit was used to develop an interactive web application for predictions.
Features:
Sliders allowed users to input values for the selected features.
The app displayed real-time predictions indicating whether the tumor was benign or malignant.

# 3. Results

## 3.1 Feature Selection
The SelectKBest method reduced the dimensionality of the dataset, retaining only the 10 most significant features. This simplification improved computational efficiency and interpretability without compromising model accuracy.

## 3.2 Model Performance
Accuracy:
The ANN achieved an accuracy of 94%.
Classification Report:
Precision: 95% for malignant tumors, 93% for benign tumors.
Recall: 96% for malignant tumors, 92% for benign tumors.
F1-Score:  95% for malignant tumors, 92% for benign tumors.
Confusion Matrix:
True Positives (Malignant): 194.
True Negatives (Benign): 342.
False Positives: 15.
False Negatives: 18.

## 3.3 Application Usability
The Streamlit application provided:

Intuitive Interface:
Users could adjust feature values using sliders.
Real-Time Feedback:
Predictions were displayed instantly, offering a seamless experience.

# 4. Discussion
## 4.1 Key Findings
Feature selection reduced model complexity while maintaining high performance.
The ANN captured non-linear patterns effectively, leading to accurate predictions.

## 4.2 Challenges
Class Imbalance:
The dataset contained more Malignant cases, which could bias the model.
Precision and recall metrics were used to evaluate performance on both classes.
Feature Engineering:
Selecting the optimal number of features required experimentation and validation.

## 4.3 Practical Implications
This project demonstrates how machine learning can enhance breast cancer diagnosis by providing a tool that is:
Efficient: Automates the diagnostic process, saving time.
Accessible: The web app makes the model’s predictions available to non-technical users.

# 5. Future Work
Validation on External Data:
Test the model on datasets from other sources to ensure generalizability.
Additional Models:
Compare ANN performance with other models, such as Random Forest or Gradient Boosting.
App Features:
Add visualizations, such as feature importance graphs, to enhance interpretability.
Include confidence intervals for predictions.

# 6. Conclusion
This project successfully developed a machine learning model for breast cancer prediction, achieving a high accuracy of 94%. The use of feature selection reduced complexity without sacrificing performance. Additionally, the Streamlit application bridges the gap between model development and practical usage, offering a simple yet powerful tool for clinicians and researchers. Future work will focus on improving generalizability and expanding the application’s capabilities.

## References
Scikit-learn Documentation: https://scikit-learn.org/
Streamlit Documentation: https://streamlit.io/
Breast Cancer Dataset (UCI Machine Learning Repository): https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)



**Appendix**
## A. Visualizations

1. Heatmap of Feature Correlations:
![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/Heatmap.png)

2. Correlation of each feature with the target variable
![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/Correlation.png)

3. Count for target values
![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/count.png)

## B. Code Snippets
1. Feature Selection

![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/SelectFeatures.png)

2. Hyper parameter tuning and Model Training

![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/HPandModel.png)

3. Streamlit App

![alt text](https://github.com/aishwaryakarki001/Breast-Cancer-Data-Analysis-and-Streamlit-App/blob/main/Images/Streamlit.png)
