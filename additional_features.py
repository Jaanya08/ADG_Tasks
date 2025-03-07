import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display basic information and summary statistics
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # Family size feature
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)  # Whether passenger is alone

# Data Preprocessing
df_ml = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handling missing values
imputer = SimpleImputer(strategy="median")
df_ml["Age"] = imputer.fit_transform(df_ml[["Age"]])
df_ml["Embarked"].fillna(df_ml["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
df_ml["Sex"] = LabelEncoder().fit_transform(df_ml["Sex"])
df_ml = pd.get_dummies(df_ml, columns=["Embarked"], drop_first=True)

# Split data
X = df_ml.drop(columns=["Survived"])
y = df_ml["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(X_train[["Age", "Fare", "FamilySize"]])
X_test[["Age", "Fare", "FamilySize"]] = scaler.transform(X_test[["Age", "Fare", "FamilySize"]])

# Compare multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)

# Best tuned model
best_model = gs.best_estimator_
y_pred_best = best_model.predict(X_test)

results["Tuned Random Forest"] = {
    "Accuracy": accuracy_score(y_test, y_pred_best),
    "Precision": precision_score(y_test, y_pred_best),
    "Recall": recall_score(y_test, y_pred_best),
    "F1 Score": f1_score(y_test, y_pred_best)
}

# Print results
print("Model Comparison:")
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
