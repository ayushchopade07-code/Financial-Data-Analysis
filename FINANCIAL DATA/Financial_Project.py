# Import all the libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import snowflake.connector

df = pd.read_csv(r"C:\Snowflake\FINANCIAL DATA\Financial Statements.csv")

df.shape
df.info()

print(df.head())

print(df.columns)

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.nunique())

# 3. DATA CLEANING

df = df.drop_duplicates()

print(df.isnull().sum())

if "Market Cap(in B USD)" in df.columns:
    df["Market Cap(in B USD)"] = df["Market Cap(in B USD)"].fillna(
        df["Market Cap(in B USD)"].median()
    )

print("\nMissing values after treatment:")
print(df.isnull().sum())

if "Category" in df.columns:
    df["Category"] = df["Category"].str.strip().str.upper()

print(df["Category"].value_counts())

# 4. UNIVARIATE ANALYSIS
# 4A. UNIVARIATE - CATEGORICAL

plt.figure(figsize=(12, 6))

df["Company "].value_counts().plot(kind="bar")

plt.title("Number of Records by Company")
plt.xlabel("Company ")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))

df["Category"].value_counts().plot(kind="bar")

plt.title("Company Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 4B. UNIVARIATE - NUMERICAL

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(df[column], kde=True)

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# 6. OUTLIER ANALYSIS

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.tight_layout()
    plt.show()

# 5. BIVARIATE ANALYSIS
# 5A. REVENUE VS NET INCOME

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Revenue",
    y="Net Income",
    hue="Category"
)

plt.title("Revenue vs Net Income")
plt.xlabel("Revenue")
plt.ylabel("Net Income")

plt.show()

# 5B. REVENUE VS MARKET CAP
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Revenue",
    y="Market Cap(in B USD)",
    hue="Category"
)

plt.title("Revenue vs Market Capitalization")
plt.xlabel("Revenue")
plt.ylabel("Market Cap (B USD)")

plt.show()

# 5C. EBITDA VS NET INCOME

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="EBITDA",
    y="Net Income",
    hue="Category"
)

plt.title("EBITDA vs Net Income")
plt.xlabel("EBITDA")
plt.ylabel("Net Income")

plt.show()

# REVENUE BY COMPANY

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="Company ",
    y="Revenue"
)

plt.title("Revenue by Company")
plt.xlabel("Company ")
plt.ylabel("Revenue")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NET INCOME BY COMPANY

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="Company ",
    y="Net Income"
)

plt.title("Net Income by Company")
plt.xlabel("Company ")
plt.ylabel("Net Income")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# YEARLY REVENUE TREND

yearly_revenue = df.groupby("Year")["Revenue"].sum().reset_index()

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=yearly_revenue,
    x="Year",
    y="Revenue",
    marker="o"
)

plt.title("Total Revenue Trend")
plt.xlabel("Year")
plt.ylabel("Total Revenue")

plt.show()

# MULTIVARIATE ANALYSIS

# REVENUE, NET INCOME, AND CATEGORY

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Revenue",
    y="Net Income",
    hue="Category",
    size="Market Cap(in B USD)",
    sizes=(50, 500),
    alpha=0.7
)

plt.title("Revenue vs Net Income by Category and Market Cap")
plt.xlabel("Revenue")
plt.ylabel("Net Income")

plt.show()

# ROE, ROA AND ROI

roe_roa_roi = df.groupby("Company ")[
    ["ROE", "ROA", "ROI"]
].mean().reset_index()

roe_roa_roi = roe_roa_roi.melt(
    id_vars="Company ",
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(14, 7))

sns.barplot(
    data=roe_roa_roi,
    x="Company ",
    y="Value",
    hue="Metric"
)

plt.title("Average ROE, ROA and ROI by Company")
plt.xlabel("Company ")
plt.ylabel("Percentage")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# CORRELATION ANALYSIS

correlation = df[numeric_columns].corr()

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Financial Statements - Correlation Heatmap")

plt.tight_layout()
plt.show()

# PAIRPLOT

selected_columns = [
    "Revenue",
    "Gross Profit",
    "Net Income",
    "EBITDA",
    "Market Cap(in B USD)"
]

sns.pairplot(
    df[selected_columns]
)

plt.show()

# IMPORT SNOWFLAKE CONNECTOR

import os
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("ENV FILE:", env_path)
print("ENV EXISTS:", env_path.exists())

print("USER:", os.getenv("SNOWFLAKE_USER") is not None)
print("PASSWORD:", os.getenv("SNOWFLAKE_PASSWORD") is not None)
print("ACCOUNT:", os.getenv("SNOWFLAKE_ACCOUNT") is not None)
print("WAREHOUSE:", os.getenv("SNOWFLAKE_WAREHOUSE") is not None)
print("DATABASE:", os.getenv("SNOWFLAKE_DATABASE") is not None)
print("SCHEMA:", os.getenv("SNOWFLAKE_SCHEMA") is not None)

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

print("Connected successfully!")

cursor = conn.cursor()
cursor.execute("SELECT CURRENT_USER(), CURRENT_ACCOUNT(), CURRENT_WAREHOUSE()")

print(cursor.fetchone())

cursor.close()
conn.close()

import snowflake.connector

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)
print("Snowflake connection successful!")

conn.close()

import snowflake.connector

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

print("Snowflake connection successful!")

query = """
SELECT *
FROM FINANCIAL_DB.COMPANY_DB.FINANCIAL_STATEMENTS
"""

df = pd.read_sql(query, conn)

print(df.head())
print("Shape:", df.shape)

# YEARLY_FINANCIAL_ANALYSIS

query = """
SELECT *
FROM FINANCIAL_DB.COMPANY_DB.YEARLY_FINANCIAL_ANALYSIS
"""

yearly_df = pd.read_sql(query, conn)

print(yearly_df)
print(yearly_df.shape)

# COMPANY_PERFORMANCE

query = """
SELECT *
FROM FINANCIAL_DB.COMPANY_DB.COMPANY_PERFORMANCE
"""

company_df = pd.read_sql(query, conn)

print(company_df)
print(company_df.shape)

# FINANCIAL_ANALYSIS

query = """
SELECT *
FROM FINANCIAL_DB.COMPANY_DB.FINANCIAL_ANALYSIS
"""

financial_df = pd.read_sql(query, conn)

print(financial_df.head())
print(financial_df.shape)

# Close the connection

conn.close()