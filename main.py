import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #to plot charts
import seaborn as sns #used for data visualization
import warnings #avoid warning flash
warnings.filterwarnings('ignore')

df = pd.read_csv("diabetes.csv")
# print(df.head())
print("The dimension of the dataframe: ", df.shape)
# Check there are non-null values data
print(df.info())
# The minimum values of Glucose, BloodPressure, SkinThickness, Insulin and BMI cannot be 0.
# The maximum value of Insulin is 846.000000.
print(df.describe())


# Data cleaning

# Drop duplicate value
df = df.drop_duplicates()
print(pd.isnull(df).sum())

print(df[df['BloodPressure'] == 0].shape[0])
print(df[df['Glucose'] == 0].shape[0])
print(df[df['SkinThickness'] == 0].shape[0])
print(df[df['Insulin'] == 0].shape[0])
print(df[df['BMI'] == 0].shape[0])

#replacing 0 values with median or mean of that column
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())#normal distribution
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())#normal distribution
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())#skewed distribution
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())#skewed distribution
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())#skewed distribution

print(df.describe())

