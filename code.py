import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("titanic_sample.csv")

# --------------------------
# 1. Basic Info and Cleaning
# --------------------------
print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing 'Age' with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Drop 'Cabin' due to too many missing values
df.drop(columns=["Cabin"], inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# --------------------------
# 2. EDA
# --------------------------

# Plotting Survival Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

# Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.show()
