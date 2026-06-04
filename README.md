# Customer-Value-Churn-Optimization-Marketing-Sales-Analytics-
Business Analytics Project: E-Commerce Customer Retention Engine  
# =====================================================================
# 1. SETUP & SIMULATED DATA GENERATION
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Setting random seed for reproducibility
np.random.seed(42)

print("⏳ Step 1: Generating simulated E-commerce transactional data...")

# Generate baseline transactional data for 1,000 customers over 1 year
num_customers = 1000
num_transactions = 15000

customer_ids = [f"CUST-{i:04d}" for i in range(1, num_customers + 1)]
states = ['California', 'New York', 'Texas', 'Florida', 'Illinois']
customer_segments_geo = np.random.choice(states, size=num_customers, p=[0.3, 0.2, 0.2, 0.15, 0.15])

cust_df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Geography': customer_segments_geo
})

# Generate random transaction dates spread across 2025
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

tx_data = []
for _ in range(num_transactions):
    cust = np.random.choice(customer_ids)
    tx_date = start_date + timedelta(days=np.random.randint(0, 365))
    amount = round(np.random.exponential(scale=75.0) + 10, 2) # Right-skewed transaction sizes
    quantity = np.random.randint(1, 10)
    tx_data.append([cust, tx_date, amount, quantity])

df = pd.DataFrame(tx_data, columns=['CustomerID', 'TransactionDate', 'OrderValue', 'Quantity'])
df = df.merge(cust_df, on='CustomerID', how='left')

# Set a snapshot date for analysis (e.g., Jan 1, 2026)
snapshot_date = datetime(2026, 1, 1)
print(f"✅ Data generated. Total Records: {df.shape[0]} across {df['CustomerID'].nunique()} unique customers.")

# =====================================================================
# 2. RFM (RECENCY, FREQUENCY, MONETARY) FEATURE ENGINEERING
# =====================================================================
print("\n⏳ Step 2: Extracting RFM Features...")

rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'TransactionDate': lambda x: x.nunique(),                   # Frequency (unique days shopped)
    'OrderValue': 'sum'                                         # Monetary Value
}).reset_index()

# Rename columns appropriately
rfm.columns = ['CustomerID', 'Frequency', 'Monetary']
# Recency needs a separate calculation because we combined it in the agg shorthand
recency_df = df.groupby('CustomerID').apply(lambda x: (snapshot_date - x['TransactionDate'].max()).days, include_groups=False).reset_index(name='Recency')
rfm = rfm.merge(recency_df, on='CustomerID')

# Reorder columns
rfm = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary']]

# Calculate RFM Scores (1 to 4 scale using quartiles)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]) # Lower recency day count = Better customer (4)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

# Combine scores to segment users
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Define Customer Tier Profiles based on R and F scores
def assign_tier(row):
    if row['R_Score'] >= 3 and row['F_Score'] >= 3:
        return 'Champions / Loyal'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At Risk / Can\'t Lose Them'
    elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
        return 'New / Recent Onboard'
    else:
        return 'Hibernating / Lost'

rfm['Customer_Tier'] = rfm.apply(assign_tier, axis=1)
print("✅ RFM Profile segments created successfully.")

# =====================================================================
# 3. CHURN LABEL DEFINITION & ML DATA PREPARATION
# =====================================================================
print("\n⏳ Step 3: Mapping Churn Labels & preparing ML features...")

# Business logic rule: If a customer hasn't purchased in the last 90 days, label them as Churned (1), else (0)
rfm['Churn'] = np.where(rfm['Recency'] > 90, 1, 0)

# Merge back geographic context data
ml_base = rfm.merge(cust_df, on='CustomerID', how='left')

# Encode Categorical variables (Geography)
ml_base = pd.get_dummies(ml_base, columns=['Geography'], drop_first=True)

# Select features for our ML Model
X = ml_base[['Recency', 'Frequency', 'Monetary', 'Geography_Florida', 'Geography_Illinois', 'Geography_New York', 'Geography_Texas']]
y = ml_base['Churn']

# Stratified Split to maintain proportional class distributions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 4. MODEL TRAINING & PERFORMANCE EVALUATION
# =====================================================================
print("\n⏳ Step 4: Training Random Forest Classifier...")

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== MODEL PERFORMANCE REPORT ===")
print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report Summary:")
print(classification_report(y_test, y_pred))

# =====================================================================
# 5. FEATURE IMPORTANCE EXTRACTION
# =====================================================================
print("\n⏳ Step 5: Extracting Feature Importance drivers...")
importances = model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print(feature_imp_df)

# =====================================================================
# 6. EXPORTING ANALYTICAL OUTPUTS FOR BUSINESS DASHBOARDS
# =====================================================================
print("\n⏳ Step 6: Saving refined dataset outputs to local working directory...")
ml_base.to_csv("ecommerce_customer_analytics.csv", index=False)
print("💾 File saved successfully as: 'ecommerce_customer_analytics.csv'")
print("\n🎉 Project Run Complete. Ready for structural documentation review.")
