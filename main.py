import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Simulate customer transaction data
num_customers = 500
num_transactions = 1000
data = {
    'CustomerID': np.random.choice(range(1, num_customers + 1), size=num_transactions),
    'TransactionDate': pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=num_transactions)),
    'TransactionAmount': np.random.exponential(scale=50, size=num_transactions),
    'Frequency': np.random.poisson(lam=2, size=num_customers)
}
df = pd.DataFrame(data)
# Aggregate data to customer level
customer_data = df.groupby('CustomerID').agg({'TransactionAmount': 'sum', 'TransactionDate': 'count'})
customer_data = customer_data.rename(columns={'TransactionDate': 'TransactionCount'})
# Feature Engineering: Recency, Frequency, Monetary Value (RFM)
max_date = df['TransactionDate'].max()
customer_data['Recency'] = (max_date - df.groupby('CustomerID')['TransactionDate'].max()).dt.days
# --- 2. Latent Class Segmentation ---
# Apply Gaussian Mixture Model for customer segmentation
gmm = GaussianMixture(n_components=3, random_state=42) #Experiment with different numbers of components
customer_data['Segment'] = gmm.fit_predict(customer_data[['Recency', 'TransactionCount', 'TransactionAmount']])
# --- 3. Analysis and Visualization ---
# Analyze segment characteristics
segment_stats = customer_data.groupby('Segment').agg({'Recency': 'mean', 'TransactionCount': 'mean', 'TransactionAmount': 'mean'})
print("Segment Statistics:")
print(segment_stats)
# Visualize segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='TransactionAmount', hue='Segment', data=customer_data, palette='viridis')
plt.title('Customer Segments based on Recency and Monetary Value')
plt.xlabel('Recency (Days)')
plt.ylabel('Total Transaction Amount')
plt.savefig('customer_segments.png')
print("Plot saved to customer_segments.png")
plt.figure(figsize=(10,6))
sns.boxplot(x='Segment', y='TransactionAmount', data=customer_data)
plt.title('Transaction Amount Distribution across Segments')
plt.savefig('transaction_amount_boxplot.png')
print("Plot saved to transaction_amount_boxplot.png")
# Identify high-risk segment (e.g., segment with highest recency and lowest monetary value)
#This would require further business understanding to define "high-risk"
high_risk_segment = segment_stats.loc[segment_stats['Recency'].idxmax()]
print("\nCharacteristics of the segment with the highest recency:")
print(high_risk_segment)