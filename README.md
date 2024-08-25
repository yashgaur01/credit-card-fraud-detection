Credit Card Fraud Detection

Overview

This project focuses on detecting fraudulent credit card transactions using advanced machine learning techniques. Given the highly imbalanced nature of the dataset, where fraudulent transactions are rare, this project aims to develop a robust model that accurately identifies fraudulent activities while minimizing false positives.

Table of Contents

Project Overview
Dataset
Preprocessing
Modeling
Evaluation
Results
Installation
Usage
Contributing
License
Acknowledgments
Dataset

The dataset used in this project is from Kaggle. It contains data on credit card transactions, with a mix of normal and fraudulent transactions. The dataset includes various features that describe the transactions and labels indicating whether the transaction is fraudulent.

Preprocessing

Handling Imbalance: Due to the dataset's imbalanced nature, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to balance the class distribution.
Feature Engineering: The dataset's features were analyzed and transformed to enhance model performance.
Data Splitting: The dataset was divided into training and testing sets using an 80-20 split.
Modeling

Multiple machine learning algorithms were implemented and evaluated, including:

Logistic Regression
Decision Trees
Random Forest
Support Vector Machines (SVM)
XGBoost
Stacking Classifier
The Stacking Classifier, which combines several models, was selected as the final model due to its superior performance.

Evaluation

The models were assessed using the following metrics:

Accuracy: The overall correctness of the model.
Precision: The proportion of positive identifications that were actually correct.
Recall: The proportion of actual positives that were identified correctly.
F1 Score: The harmonic mean of precision and recall.
Results

The final model, utilizing a Stacking Classifier, achieved the following results on the test set:

Accuracy: 99.64%
Precision: 70.56%
Recall: 63.98%
F1 Score: 67.11%
These metrics demonstrate the model's effectiveness in identifying fraudulent transactions, with room for further optimization.

Installation

To replicate this project locally, ensure you have Python installed along with the required libraries:

bash
Copy code
pip install numpy pandas scikit-learn imbalanced-learn xgboost
Clone this repository and run the Jupyter Notebook:

bash
Copy code
git clone https://github.com/Yashgaur1/Credit-Card_Fraud-Detection_Project.git
cd Credit-Card_Fraud-Detection_Project
Usage

Run the Jupyter Notebook CCFD-2.ipynb to explore the preprocessing steps, model training, and evaluation process.

Contributing

Contributions to this project are welcome! If you have ideas for improvements or want to add new features, feel free to fork the repository and submit a pull request.

License

This project is licensed under the MIT License. For more details, see the LICENSE file.

Acknowledgments

The dataset was sourced from Kaggle and can be accessed here.
This project was inspired by the ongoing need to develop better tools for detecting and preventing financial fraud.
