import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r"C:\Users\chand\Downloads\youtube_recommendation_dataset -.csv")
# EDA
df.head()
df.info()
df.describe()
df.shape
print("\nMissing Values:")
print(df.isnull().sum())

df.columns

fig, axes = plt.subplots(2, 2, figsize=(15, 20))
fig.suptitle('Distribution of Key Metrics', fontsize=16)

sns.histplot(df['view_count'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribution of View Count')
axes[0, 0].set_xlabel('Views')

sns.histplot(df['like_count'], bins=30, kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Distribution of Like Count')
axes[0, 1].set_xlabel('Likes')

sns.histplot(df['comment_count'], bins=30, kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Distribution of Comment Count')
axes[1, 0].set_xlabel('Comments')

sns.histplot(df['duration_seconds'], bins=30, kde=True, ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Distribution of Duration (Seconds)')
axes[1, 1].set_xlabel('Duration (s)')
axes[1, 1].set_xlim(0, df['duration_seconds'].quantile(0.95)) # Limit x-axis for better visibility

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig, axes = plt.subplots(1, 2, figsize=(15, 10))

sns.countplot(x='category_id', data=df, ax=axes[0], palette='viridis', order=df['category_id'].value_counts().index)
axes[0].set_title('Video Count by Category ID')

sns.countplot(x='caption', data=df, ax=axes[1], palette='pastel')
axes[1].set_title('Videos with Captions')

plt.tight_layout()

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')

fig, axes = plt.subplots(1, 2, figsize=(15, 10))

sns.scatterplot(x='view_count', y='like_count', data=df, ax=axes[0], alpha=0.6)
axes[0].set_title('Views vs. Likes')
axes[0].set_xscale('log') # Log scale for better visualization
axes[0].set_yscale('log')

sns.scatterplot(x='view_count', y='comment_count', data=df, ax=axes[1], alpha=0.6, color='orange')
axes[1].set_title('Views vs. Comments')
axes[1].set_xscale('log')
axes[1].set_yscale('log')

plt.tight_layout()

plt.figure(figsize=(15, 8))
sns.boxplot(x='category_id', y='view_count', data=df, palette='Set2')
plt.yscale('log')
plt.title('View Count Distribution by Category')
