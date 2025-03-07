import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
