# ============================
# NexaTel Customer Churn Analytics - EDA Skeleton with Chart Saving
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Notebook settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# Ensure charts folder exists
os.makedirs("charts", exist_ok=True)

# Load Data
customers = pd.read_csv("Cleaned nexa data/customers.csv")
subscriptions = pd.read_csv("Cleaned nexa data/subscriptions.csv")
billing = pd.read_csv("Cleaned nexa data/billing.csv")
usage_data = pd.read_csv("Cleaned nexa data/usage_data.csv")

# Customer Age Distribution
sns.histplot(customers['age'], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.savefig("charts/customer_age_distribution.png")
plt.close()

# Gender Split
sns.countplot(x='gender', data=customers)
plt.title("Gender Split")
plt.savefig("charts/gender_split.png")
plt.close()

# Subscription Status
sns.countplot(x='subscription_status', data=subscriptions)
plt.title("Subscription Status Distribution")
plt.savefig("charts/subscription_status_distribution.png")
plt.close()

# Billing Cycle
sns.histplot(billing['total_amount_inr'], bins=30, kde=True)
plt.title("Billing Amount Distribution")
plt.savefig("charts/billing_amount_distribution.png")
plt.close()

# Churn Rate
churn_rate = (subscriptions['subscription_status']=='Terminated').mean()
print("Churn Rate:", round(churn_rate*100,2), "%")

# Usage Distribution
sns.histplot(usage_data['data_gb_used'], bins=30, kde=True)
plt.title("Data Usage Distribution")
plt.savefig("charts/data_usage_distribution.png")
plt.close()

# Outlier vs Normal Usage
sns.boxplot(x='usage_status', y='data_gb_used', data=usage_data)
plt.title("Outlier vs Normal Usage")
plt.savefig("charts/outlier_vs_normal_usage.png")
plt.close()

# Billing Amount Distribution
sns.histplot(billing['total_amount_inr'], bins=30, kde=True)
plt.title("Billing Amount Distribution")
plt.savefig("charts/billing_amount_distribution.png")
plt.close()

# Payment Status
sns.countplot(x='payment_status', data=billing)
plt.title("Payment Status Distribution")
plt.savefig("charts/payment_status_distribution.png")
plt.close()

# Correlation Heatmap
numeric_cols = usage_data.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("charts/correlation_heatmap.png")
plt.close()
