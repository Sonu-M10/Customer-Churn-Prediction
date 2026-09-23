# Customer Churn Prediction

## Project Goal

This project uses machine learning to predict customer churn for a telecommunications company.

The goal is to identify customers who are more likely to leave so that businesses can better understand churn patterns and potentially take action to improve customer retention.

## Dataset

This project uses the IBM Telco Customer Churn dataset.

The original dataset contains 7,043 customer records. After cleaning the `TotalCharges` column and removing rows with missing values, 7,032 records remained.

## Data Preparation

The data was prepared using the following steps:

1. Loaded the dataset using Pandas
2. Converted `TotalCharges` from text to numeric values
3. Removed rows with missing values
4. Removed the customer ID from the modeling features
5. Converted categorical variables using one-hot encoding
6. Split the data into training and testing sets
7. Standardized the features using `StandardScaler`

The final dataset contained 30 features.

## Models Tested

Several machine learning models were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Voting Ensemble

Logistic Regression was selected as the final model because it provided strong test performance while also being easier to interpret than the more complex models tested.

## Results

The final Logistic Regression model achieved:

- **Cross-validation accuracy:** 80.30%
- **Test accuracy:** 79.82%

The model was tuned using `GridSearchCV`.

## Threshold Analysis

The project also tested different classification thresholds.

A lower threshold identifies more customers who actually churn, but it also creates more false positives.

For example, at a threshold of 0.40:

- Accuracy: 78.68%
- Churn recall: 68%

At the default threshold of 0.50:

- Accuracy: 79.82%
- Churn recall: 53%

This shows the tradeoff between overall accuracy and identifying customers who are likely to churn.

## Important Features

Some of the features with stronger model coefficients included:

- Tenure
- Contract type
- Fiber optic internet service
- Monthly charges
- Online security
- Tech support
- Payment method

These coefficients represent associations learned by the model and do not necessarily mean that a feature directly causes customer churn.

## Project Visualizations

The project includes visualizations showing:

- Customer churn distribution
- Model accuracy comparison
- Classification threshold tradeoffs

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Git
- GitHub

## What I Learned

This project helped me practice the full machine learning workflow, from cleaning and exploring data to preparing features, training models, tuning hyperparameters, evaluating results, and interpreting model predictions.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── cleaned_churn.csv
│
├── graphs/
│   ├── churn_distribution.png
│   ├── model_comparison.png
│   └── threshold_tradeoff.png
│
├── SRC/
│   ├── model.py
│   └── model1.py
│
├── notebooks/
│
├── .gitignore
└── README.md
```
