''''''
# this is just a rough work in creating the final work


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


#plt.plot([1, 2, 3], [4, 5, 6])

# Open the graph without stopping the code
#plt.show(block=False) 
#plt.pause(0.1) # Essential to let the window render

# Your script will immediately continue down here!
#print("This message prints while the graph is still open!")

#data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data = pd.read_csv("data/cleaned_churn.csv")


print(data.head())
print(data.shape)
print(data.info())

print(data.isnull().sum())
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

print(data.dtypes)

data = data.dropna()
data["AverageMonthlySpend"] = data["TotalCharges"] / data["tenure"].replace(0, 1)

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
print(data.isnull().sum())

# More feature engineering

data["TenureSquared"] = data["tenure"] ** 2

data["MonthlyChargesSquared"] = data["MonthlyCharges"] ** 2



data["IsNewCustomer"] = (data["tenure"] <= 6).astype(int)

data["IsLongTermCustomer"] = (data["tenure"] >= 48).astype(int)

data["HighMonthlyCharges"] = (
    data["MonthlyCharges"] >= data["MonthlyCharges"].median()
).astype(int)

data["HasSecuritySupport"] = (
    (data["OnlineSecurity"] == "Yes")
    | (data["TechSupport"] == "Yes")
).astype(int)

data["HasStreaming"] = (
    (data["StreamingTV"] == "Yes")
    | (data["StreamingMovies"] == "Yes")
).astype(int)

# for reference print(data["TotalCharges"])

data.to_csv("data/cleaned_churn.csv", index=False)


#data["Churn"].value_counts().plot(kind="bar")

#plt.title("Customer Churn")
#plt.xlabel("Churn")
#plt.ylabel("Number of Customers")

#plt.show()

#if contract type is related to churn.

#print(pd.crosstab(data["Contract"], data["Churn"]))

#pd.crosstab(data["Contract"], data["Churn"]).plot(kind="bar")

#plt.title("Churn by Contract Type")
#plt.xlabel("Contract Type")
#plt.ylabel("Number of Customers")

#plt.show()

print(data["tenure"].describe())

#Calculate churn rate by contract

churn_rate = pd.crosstab(
    data["Contract"],
    data["Churn"],
    normalize="index"
) * 100
print (churn_rate)
print(churn_rate.round(2).astype(str) + "%")


#Compare tenure between churned and non-churned customers

#print(data.groupby("Churn")["tenure"].mean())
#data.groupby("Churn")["tenure"].mean().plot(kind="bar")

#plt.title("Average Tenure by Churn")
#plt.xlabel("Churn")
#plt.ylabel("Average Tenure (Months)")

#plt.show()

#Visualize monthly charges

#print(data.groupby("Churn")["MonthlyCharges"].mean())
#data.groupby("Churn")["MonthlyCharges"].mean().plot(kind="bar")

#plt.title("Average Monthly Charges by Churn")
##plt.xlabel("Churn")
#plt.ylabel("Average Monthly Charges")

#plt.show()

print(data.dtypes)

print(data.dtypes)


print(data.select_dtypes(include="str").columns)

#removing customer id as it is useless for our ML
data = data.drop("customerID", axis=1)

X = data.drop("Churn", axis=1)

engineered_features = [
    "AverageMonthlySpend",
    "ServiceCount",
    "TenureMonthlyCharge",
    "TenureContract",
    "HighChargeShortTenure",
    "ElectronicCheckMonthToMonth",
    "TenureSquared",
    "MonthlyChargesSquared",
    "IsNewCustomer",
    "IsLongTermCustomer",
    "HighMonthlyCharges",
    "HasSecuritySupport",
    "HasStreaming"
]



X = X.drop(engineered_features, axis=1)
y = data["Churn"]

#encoding for the ML

print(X.head())
print(y.head())

X = pd.get_dummies(X, drop_first=True)

y = y.map({"No": 0, "Yes": 1})

print(X.head())
print(y.head())


#splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print(X_train.shape)
print(X_test.shape)


#creating the ML with the data 

for c in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("C =", c)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))




# Get churn probabilities from Logistic Regression

from sklearn.model_selection import GridSearchCV

logistic = LogisticRegression(max_iter=5000)

param_grid = {
    "C": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100],
    "solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    logistic,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)


# Test different churn probability thresholds

model = grid_search.best_estimator_
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    threshold_pred = (y_prob >= threshold).astype(int)
    print("Threshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, threshold_pred))
    print(classification_report(y_test, threshold_pred))
    print(confusion_matrix(y_test, threshold_pred))

#bottleneck at 80.3-80.5% accuracy good for rough work

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False
)

print(coefficients)


#decision tree

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))
print(confusion_matrix(y_test, tree_pred))


#adding a random forest model

# Random Forest GridSearchCV

rf = RandomForestClassifier(random_state=42)

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    rf,
    rf_param_grid,
    cv=5,
    scoring="accuracy"
)

rf_grid.fit(X_train, y_train)

print("Best Random Forest Parameters:", rf_grid.best_params_)
print("Best Random Forest Cross-Validation Accuracy:", rf_grid.best_score_)

rf_best = rf_grid.best_estimator_

rf_best_pred = rf_best.predict(X_test)

print("Tuned Random Forest Test Accuracy:", accuracy_score(y_test, rf_best_pred))
print(classification_report(y_test, rf_best_pred))
print(confusion_matrix(y_test, rf_best_pred))


#adding a LDA model

lda_model = LinearDiscriminantAnalysis()

lda_model.fit(X_train, y_train)

lda_pred = lda_model.predict(X_test)

print("LDA Accuracy:", accuracy_score(y_test, lda_pred))
print(classification_report(y_test, lda_pred))
print(confusion_matrix(y_test, lda_pred))


# adding a knn model
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))


# Tuned Gradient Boosting
gradient_model = GradientBoostingClassifier(random_state=42)

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

gradient_grid.fit(X_train, y_train)

print("Best Gradient Boosting Parameters:", gradient_grid.best_params_)
print("Best Gradient Boosting Cross-Validation Accuracy:", gradient_grid.best_score_)

gradient_model = gradient_grid.best_estimator_
gradient_pred = gradient_model.predict(X_test)

print("Tuned Gradient Boosting Test Accuracy:", accuracy_score(y_test, gradient_pred))
print(classification_report(y_test, gradient_pred))
print(confusion_matrix(y_test, gradient_pred))


#creating a voting ensemble with the 5 models

voting_model = VotingClassifier(
    estimators=[
        ("logistic", LogisticRegression(max_iter=5000)),
        ("lda", LinearDiscriminantAnalysis()),
        ("tree", DecisionTreeClassifier(random_state=42)),
        ("forest", RandomForestClassifier(random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ],
    voting="soft"
)

voting_model.fit(X_train, y_train)

voting_pred = voting_model.predict(X_test)

print("Voting Ensemble Accuracy:", accuracy_score(y_test, voting_pred))
print(classification_report(y_test, voting_pred))
print(confusion_matrix(y_test, voting_pred))


''''''

