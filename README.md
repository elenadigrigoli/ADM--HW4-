# **ADM--HW4**
# **Movie Recommendation System**
ADM | Sapienza University of Rome
This repository contains the solution to Homework 4, which focuses on implementing a Movie Recommendation System and performing clustering of movies based on their features.

#### **Point 1: Recommendation System with LSH**
- Implement a recommendation system based on user similarities.
- Use MinHash and Locality-Sensitive Hashing (LSH) to identify similar users.
- Evaluate the precision of the recommendation system.
  
**What we did**:
- Preprocessed the data, handling missing values, duplicates, and inconsistent formats.
- Created MinHash signatures for each user and estimated Jaccard Similarity.
- Divided the signatures into "bands" and created "buckets" using LSH to group similar users.
- Calculated similarity between user pairs using MinHash signatures.
- Recommended movies based on highly-rated films from the most similar users.
- Calculate Precision, Recall, and F1-score to measure the quality of recommendations.
- Measured performance across different similarity thresholds, evaluating the balance between precision and recall.

---

#### **Point 2: Grouping Movies Together!**
- Create new features
- Group movies based on features like genres, ratings, and tags.
- Apply clustering methods such as KMeans and DBSCAN+
- Evaluate clustering results using the Elbow Method and Silhouette Score and other methods.
-  Determine the optimal number of clusters.
- Compare the results of clustering methods (KMeans, KMeans++, DBSCAN).
- Visualize the results to explain cluster distributions.

**What we did**:
- Normalized numeric data.
- Reduced dimensionality using.
- Applied:
  - **KMeans** to partition movies into clusters.
  - **KMeans++** to improve centroid initialization.
  - **DBSCAN** to detect dense clusters and outliers.
  -  Used the Elbow Method to estimate the optimal number of clusters.
  - Calculated the Silhouette Score to evaluate cluster cohesion.
  - Applied the k-Distance Graph to select the optimal epsilon value for DBSCAN.


---
#### **Point 3: Bonus Question**
- The task focuses on visualizing the iterative process of the K-means clustering algorithm. As K-means refines clusters with each iteration, this bonus task aims to illustrate how the clusters evolve over time.

- **What We Did**:
- Tracked clustering progress over 10 iterations, visualizing data points and centroids in a 2D space while observing centroid movements and explained variance.
 

---

#### **Algorithm Question**

- Find Arya's optimal strategy for maximizing her score.
- Analyze the algorithm's efficiency and time complexity.
- Optimize it (if needed) and compare results with the original.
- Use an LLM to suggest improvements and evaluate its solution.


