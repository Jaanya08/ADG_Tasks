import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display basic information and summary statistics
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Survival distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title("Survival Distribution (0 = No, 1 = Yes)")
plt.show()

# Survival rate by passenger class
plt.figure(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Passenger Class")
plt.show()

# Survival rate by gender
plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Gender")
plt.show()

# Age distribution by survival
plt.figure(figsize=(8,5))
sns.histplot(df[df['Survived'] == 1]['Age'], bins=30, kde=True, color='green', label='Survived')
sns.histplot(df[df['Survived'] == 0]['Age'], bins=30, kde=True, color='red', label='Did not Survive')
plt.title("Age Distribution by Survival")
plt.legend()
plt.show()

# Relationship between Fare and Survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=df, palette='coolwarm')
plt.title("Fare vs. Survival")
plt.show()

# Survival rate based on Embarked location
plt.figure(figsize=(6,4))
sns.barplot(x='Embarked', y='Survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Embarkation Point")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

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
X_train[["Age", "Fare"]] = scaler.fit_transform(X_train[["Age", "Fare"]])
X_test[["Age", "Fare"]] = scaler.transform(X_test[["Age", "Fare"]])

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
