"""
@author Moresh Shukla
Customer Churn Prediction
"""
import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load and clean the data
# --------------------------------------------------

data = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Original data:")
print(data.head())
print("Shape:", data.shape)

# Convert TotalCharges from text to numbers
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
data = data.dropna()

print("\nMissing values:")
print(data.isnull().sum())

print("\nCleaned shape:", data.shape)


# --------------------------------------------------
# 2. Explore the data
# --------------------------------------------------

print("\nTenure statistics:")
print(data["tenure"].describe())

print("\nChurn rate by contract:")

churn_rate = pd.crosstab(
    data["Contract"],
    data["Churn"],
    normalize="index"
) * 100

print(
    churn_rate.round(2).astype(str) + "%"
)


# --------------------------------------------------
# 3. Feature engineering experiments
# --------------------------------------------------

data["AverageMonthlySpend"] = (
    data["TotalCharges"]
    / data["tenure"].replace(0, 1)
)

data["ServiceCount"] = (
    (data["PhoneService"] == "Yes").astype(int)
    + (data["MultipleLines"] == "Yes").astype(int)
    + (data["OnlineSecurity"] == "Yes").astype(int)
    + (data["OnlineBackup"] == "Yes").astype(int)
    + (data["DeviceProtection"] == "Yes").astype(int)
    + (data["TechSupport"] == "Yes").astype(int)
    + (data["StreamingTV"] == "Yes").astype(int)
    + (data["StreamingMovies"] == "Yes").astype(int)
)

# These features were tested during development.
# They did not provide a meaningful improvement,
# so they are not used in the final model.


# --------------------------------------------------
# 4. Prepare features and target
# --------------------------------------------------

# Customer ID does not provide useful predictive information
data = data.drop(
    "customerID",
    axis=1
)

X = data.drop(
    "Churn",
    axis=1
)

y = data["Churn"]

# Remove experimental features from final model
X = X.drop(
    [
        "AverageMonthlySpend",
        "ServiceCount"
    ],
    axis=1
)

# Convert categorical variables into numerical columns
X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert target to 0/1
y = y.map({
    "No": 0,
    "Yes": 1
})

print(
    "\nFinal feature count:",
    X.shape[1]
)


# --------------------------------------------------
# 5. Split and scale the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(
    "Training set:",
    X_train.shape
)

print(
    "Testing set:",
    X_test.shape
)


# --------------------------------------------------
# 6. Logistic Regression
# --------------------------------------------------

print("\n--- Logistic Regression ---")

for c in [0.01, 0.1, 1, 10, 100]:

    logistic_model = LogisticRegression(
        C=c,
        max_iter=5000
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    predictions = logistic_model.predict(
        X_test
    )

    print("\nC =", c)

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )


# --------------------------------------------------
# 7. Tune Logistic Regression
# --------------------------------------------------

print("\n--- Logistic Regression Grid Search ---")

logistic = LogisticRegression(
    max_iter=5000
)

logistic_param_grid = {
    "C": [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10,
        25,
        50,
        100
    ],
    "solver": [
        "liblinear",
        "lbfgs"
    ]
}

logistic_grid = GridSearchCV(
    logistic,
    logistic_param_grid,
    cv=5,
    scoring="accuracy"
)

logistic_grid.fit(
    X_train,
    y_train
)

print(
    "Best Parameters:",
    logistic_grid.best_params_
)

print(
    "Best Cross-Validation Accuracy:",
    logistic_grid.best_score_
)

# Save the best logistic regression model
logistic_model = logistic_grid.best_estimator_

logistic_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. Threshold Analysis
# --------------------------------------------------

print("\n--- Threshold Analysis ---")

y_prob = logistic_model.predict_proba(
    X_test
)[:, 1]

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

threshold_accuracy = []
churn_recall = []

for threshold in thresholds:

    threshold_predictions = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        threshold_predictions
    )

    report = classification_report(
        y_test,
        threshold_predictions,
        output_dict=True
    )

    recall = report["1"]["recall"]

    threshold_accuracy.append(
        accuracy * 100
    )

    churn_recall.append(
        recall * 100
    )

    print(
        "\nThreshold:",
        threshold
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        classification_report(
            y_test,
            threshold_predictions
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            threshold_predictions
        )
    )


# --------------------------------------------------
# 9. Logistic Regression Coefficients
# --------------------------------------------------

print(
    "\n--- Logistic Regression Coefficients ---"
)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logistic_model.coef_[0]
})

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False
)

print(coefficients)


# --------------------------------------------------
# 10. Decision Tree
# --------------------------------------------------

print("\n--- Decision Tree ---")

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_predictions = tree_model.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print(
    "Decision Tree Accuracy:",
    tree_accuracy
)

print(
    classification_report(
        y_test,
        tree_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        tree_predictions
    )
)


# --------------------------------------------------
# 11. Random Forest
# --------------------------------------------------

print("\n--- Random Forest ---")

random_forest = RandomForestClassifier(
    random_state=42
)

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    random_forest,
    rf_param_grid,
    cv=5,
    scoring="accuracy"
)

rf_grid.fit(
    X_train,
    y_train
)

print(
    "Best Random Forest Parameters:",
    rf_grid.best_params_
)

print(
    "Best Random Forest Cross-Validation Accuracy:",
    rf_grid.best_score_
)

rf_best = rf_grid.best_estimator_

rf_predictions = rf_best.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest Test Accuracy:",
    rf_accuracy
)

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# --------------------------------------------------
# 12. Linear Discriminant Analysis
# --------------------------------------------------

print("\n--- LDA ---")

lda_model = LinearDiscriminantAnalysis()

lda_model.fit(
    X_train,
    y_train
)

lda_predictions = lda_model.predict(
    X_test
)

lda_accuracy = accuracy_score(
    y_test,
    lda_predictions
)

print(
    "LDA Accuracy:",
    lda_accuracy
)

print(
    classification_report(
        y_test,
        lda_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        lda_predictions
    )
)


# --------------------------------------------------
# 13. K-Nearest Neighbors
# --------------------------------------------------

print("\n--- KNN ---")

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train,
    y_train
)

knn_predictions = knn_model.predict(
    X_test
)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

print(
    "KNN Accuracy:",
    knn_accuracy
)

print(
    classification_report(
        y_test,
        knn_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        knn_predictions
    )
)


# --------------------------------------------------
# 14. Gradient Boosting
# --------------------------------------------------

print("\n--- Gradient Boosting ---")

gradient_model = GradientBoostingClassifier(
    random_state=42
)

gradient_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}

gradient_grid = GridSearchCV(
    gradient_model,
    gradient_param_grid,
    cv=5,
    scoring="accuracy"
)

gradient_grid.fit(
    X_train,
    y_train
)

print(
    "Best Gradient Boosting Parameters:",
    gradient_grid.best_params_
)

print(
    "Best Gradient Boosting Cross-Validation Accuracy:",
    gradient_grid.best_score_
)

gradient_best = gradient_grid.best_estimator_

gradient_predictions = gradient_best.predict(
    X_test
)

gradient_accuracy = accuracy_score(
    y_test,
    gradient_predictions
)

print(
    "Gradient Boosting Test Accuracy:",
    gradient_accuracy
)

print(
    classification_report(
        y_test,
        gradient_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        gradient_predictions
    )
)


# --------------------------------------------------
# 15. Voting Ensemble
# --------------------------------------------------

print("\n--- Voting Ensemble ---")

voting_model = VotingClassifier(
    estimators=[
        (
            "logistic",
            LogisticRegression(max_iter=5000)
        ),
        (
            "lda",
            LinearDiscriminantAnalysis()
        ),
        (
            "tree",
            DecisionTreeClassifier(
                random_state=42
            )
        ),
        (
            "forest",
            RandomForestClassifier(
                random_state=42
            )
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ],
    voting="soft"
)

voting_model.fit(
    X_train,
    y_train
)

voting_predictions = voting_model.predict(
    X_test
)

voting_accuracy = accuracy_score(
    y_test,
    voting_predictions
)

print(
    "Voting Ensemble Accuracy:",
    voting_accuracy
)

print(
    classification_report(
        y_test,
        voting_predictions
    )
)

print(
    confusion_matrix(
        y_test,
        voting_predictions
    )
)


# --------------------------------------------------
# 16. Store Model Accuracies
# --------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_model.predict(X_test)
)

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "LDA",
    "KNN",
    "Gradient Boosting",
    "Voting Ensemble"
]

model_accuracies = [
    logistic_accuracy * 100,
    tree_accuracy * 100,
    rf_accuracy * 100,
    lda_accuracy * 100,
    knn_accuracy * 100,
    gradient_accuracy * 100,
    voting_accuracy * 100
]


# --------------------------------------------------
# 17. Churn Distribution Graph
# --------------------------------------------------

churn_counts = data["Churn"].value_counts()

plt.figure(figsize=(6, 4))

plt.bar(
    ["No", "Yes"],
    [
        churn_counts["No"],
        churn_counts["Yes"]
    ]
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("graphs/churn_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 18. Threshold Analysis Graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    threshold_accuracy,
    marker="o",
    label="Accuracy"
)

plt.plot(
    thresholds,
    churn_recall,
    marker="o",
    label="Churn Recall"
)

plt.title(
    "Classification Threshold Tradeoff"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Percentage (%)"
)

plt.legend()

plt.tight_layout()
plt.savefig("graphs/threshold_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 19. Model Accuracy Comparison Graph
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    model_names,
    model_accuracies
)

plt.title(
    "Model Accuracy Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 20. Final Model
# --------------------------------------------------

print("\n--- Final Model ---")

print(
    "Final Model: Logistic Regression"
)

print(
    "Best Parameters:",
    logistic_grid.best_params_
)

print(
    "Cross-Validation Accuracy:",
    round(
        logistic_grid.best_score_ * 100,
        2
    ),
    "%"
)

print(
    "Final Test Accuracy:",
    round(
        logistic_accuracy * 100,
        2
    ),
    "%"
)

print("\nProject completed successfully.")