# Exploratory Data Analysis on Titanic Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder to save charts
if not os.path.exists("output_charts"):
    os.makedirs("output_charts")

# Load dataset
data = pd.read_csv("C:/Users/divya/Desktop/Divyam_Datascience_week1/train.csv.csv")
print("✅ Dataset Loaded Successfully\n")

# Basic information
print("--- Dataset Info ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Data Description ---")
print(data.describe())

# Handle missing and duplicate data
data.drop_duplicates(inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
print("\n✅ Missing and duplicate data handled\n")

# Visualizations
# 1. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title("Survival Count by Gender")
plt.savefig("output_charts/survival_by_gender.png")
plt.show()

# 2. Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.savefig("output_charts/survival_by_class.png")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("output_charts/age_distribution.png")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("output_charts/correlation_heatmap.png")
plt.show()

print("✅ Charts saved in 'output_charts' folder\n")

# Summary report
summary = """
--- SUMMARY REPORT ---

1. Females had a higher survival rate compared to males.
2. Passengers in 1st class were more likely to survive than those in 3rd class.
3. Most passengers were aged between 20–40 years.
4. Age and Embarked columns had missing values that were handled.
5. Clear correlation observed between survival, gender, and class.
"""

print(summary)

with open("output_charts/summary_report.txt", "w") as f:
    f.write(summary)

print("✅ Summary report saved as summary_report.txt")
