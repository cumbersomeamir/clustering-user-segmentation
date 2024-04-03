#Generating a Synthetic Dataset
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Generating synthetic data
n_samples = 1000
features, _ = make_blobs(n_samples=n_samples, centers=5, n_features=5, random_state=42)

# Adding categorical features
np.random.seed(42)
features_categorical = {
    'most_active_time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
    'user_type': np.random.choice(['New', 'Active', 'Churned'], n_samples)
}

# Convert to DataFrame
df = pd.DataFrame(features, columns=['number_of_posts', 'average_post_length', 'number_of_friends', 'average_daily_time', 'preferred_content_category'])
df = pd.concat([df, pd.DataFrame(features_categorical)], axis=1)

# Feature Engineering
# Creating an engagement rate feature (posts * average_daily_time / friends)
df['engagement_rate'] = df['number_of_posts'] * df['average_daily_time'] / np.maximum(df['number_of_friends'], 1)

# Handling categorical features with One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['number_of_posts', 'average_post_length', 'number_of_friends', 'average_daily_time', 'preferred_content_category', 'engagement_rate']),
        ('cat', OneHotEncoder(), ['most_active_time_of_day', 'user_type'])
    ])

X_processed = preprocessor.fit_transform(df)


#Advanced Clustering with K-Means
from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_processed)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Fit K-Means with the optimal number of clusters (assuming we chose 5 based on the plot)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_processed)

# Since the data is now in a transformed space, direct plotting will not be as interpretable. We can, however, analyze the clusters.

'''Analyse Cluster Centroids'''

# Assuming kmeans is your fitted KMeans model and X_processed is your processed data
centroids = kmeans.cluster_centers_

# If you have scaled your numerical data, you may want to transform centroids back to the original scale
# This is complex due to the categorical features, but for numerical features, you can use the scaler's inverse_transform method

# Print out centroids for interpretation
print(centroids)

'''Dimentionality Reduction for Visualisation'''
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA and reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title('Clusters Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()


'''3. Interpret and Act on Clusters
Profile Each Cluster: Use the cluster centroids and the PCA visualization to profile each cluster. For instance, a cluster with high engagement rates and a preference for evening activity might represent highly engaged nighttime users.
Tailor Content and Recommendations: Based on the cluster profiles, tailor your content recommendations and engagement strategies. For example, the cluster identified as nighttime users might prefer content that's more engaging during those hours.
Monitor Cluster Evolution: User preferences and behaviors can evolve. Periodically re-run your clustering analysis to capture any changes in user behavior or to identify new emerging segments.

1. Multidimensional Analysis
Simple Metric Approach: Focusing on a single metric, such as activity during a specific time period, provides a unidimensional view. This approach might tell you that a user is active in the evening, but it lacks depth. You know when they're active, but not how their activity compares in complexity (e.g., posting vs. browsing) or what content they prefer.

Clustering (Machine Learning) Approach: Clustering considers multiple features simultaneously, including the time of activity, types of engagement (likes, comments, shares), content preferences, and more. This multidimensional analysis enables the identification of nuanced user segments that share complex behavior patterns, not just a single common trait. It reveals not just when users are active, but how they interact with the platform in a holistic manner.

2. Discovering Hidden Patterns
Insight Depth: Beyond surface-level metrics, machine learning can uncover non-obvious patterns and relationships within the data. For example, clustering might reveal a segment of users who are particularly active in the evening but primarily engage with educational content—a pattern that simple usage metrics could overlook.

Predictive Power: Machine learning methods can also predict future behaviors based on historical data patterns, allowing for proactive content recommendation and engagement strategies tailored not just to current habits but anticipated needs or interests.

3. Personalization at Scale
Scalability: While it's feasible to manually segment users based on a few criteria, machine learning algorithms like clustering can handle vast datasets with numerous variables, enabling personalization at scale. This means being able to cater to the unique preferences and behaviors of millions of users efficiently.

Dynamic Adaptation: User behaviors change over time. Machine learning models can adapt to these changes, continually learning from new data to refine and update user segments. This ensures that the strategies based on these segments remain relevant and effective.
'''






