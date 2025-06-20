import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
data_path = 'data/air_quality_health_dataset.csv'
df = pd.read_csv(data_path)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

# AQI vs Hospital Admissions
plt.figure(figsize=(8, 6))
sns.scatterplot(x='aqi', y='hospital_admissions', data=df)
plt.title('AQI vs Hospital Admissions')
plt.savefig('outputs/aqi_vs_hospital_admissions.png')
plt.close()

print("✅ Visualizations saved in the 'outputs/' directory.")
