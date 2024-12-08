# %% [markdown]
# # **HW4 - Movie Recommendation System**

# %% [markdown]
# ## **1. Recommendation System with LHS** 

# %% [markdown]
# ### 1.1 Data Preparation

# %%
#!pip install datasketch
# import necessary libraries
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# %%
# Read all datasets
genome_scores = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/genome_scores.csv")
genome_tags = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/genome_tags.csv")
link = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/link.csv")
movie = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/movie.csv")
rating = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/rating.csv")
tag = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/tag.csv")


# %%
movie = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/movie.csv')
ratings = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/rating.csv')
genome_scores = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/genome_scores.csv') 
tags = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/tag.csv')
genome_tags = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/genome_tags.csv')


# %%
# Replaces genre separators ('|') in the 'genres' column with spaces.
movie['genres'] = movie['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')


# %%
# Display informations about the datasets.
print("\nInfo about 'movies':")
print(movie.info())

print("\nInfo about 'ratings':")
print(rating.info())


print("\nInfo about 'tag':")
print(tag.info())

print("\nInfo about 'genome_tags':")
print(genome_tags.info())


print("\nInfo about 'genome_scores':")
print(genome_scores.info())

print("\nInfo about 'link':")
print(link.info())

# %%
# Display basic information and the first few rows of each dataset for an initial inspection.
display(movie.head())
display(movie.head())
display(rating.head())
display(tag.head())
display(genome_tags.head())
display(genome_scores.head())
display(link.head())

# %%
# Checks for missing values in each dataset and prints the results.
print(rating.isnull().sum())
print(movie.isnull().sum())
print(tag.isnull().sum())
print(link.isnull().sum())
print(genome_scores.isnull().sum())
print(genome_tags.isnull().sum())

# %%
# Check for missing values (NaN) in the datasets and print the count for each column.
print(rating.isna().sum())
print(movie.isna().sum())
print(tag.isna().sum())
print(link.isna().sum())
print(genome_scores.isna().sum())
print(genome_tags.isna().sum())

# %%
# Replace missing values in 'tag' column with the word 'unknown'.
tag['tag'] = tag['tag'].fillna('unknown')
print(tag['tag'].isna().sum())  #Should return 0 

# %%
# Remove and display duplicate rows in the datasets.
display(rating.drop_duplicates())
display(movie.drop_duplicates())
display(tag.drop_duplicates())
display(link.drop_duplicates())
display(genome_scores.drop_duplicates())
display(genome_tags.drop_duplicates())

# %%
#Descriptive statistics for all datasets
print("\nDescriptive statistics for 'genome_scores':")
print(genome_scores.describe())

print("\nDescriptive statistics for 'genome_tags':")
print(genome_tags.describe(include='all'))  

print("\nDescriptive statistics for 'link':")
print(link.describe(include='all'))  

print("\nDescriptive statistics for 'movie':")
print(movie.describe(include='all'))  
print("\nDescriptive statistics for 'ratings':")
print(rating.describe())

print("\nDescriptive statistics for 'tag':")
print(tag.describe(include='all'))  

# %%
# Identifies movie IDs present in the 'rating' dataset but missing in the 'movie' dataset.
missing_movies = set(rating['movieId']) - set(movie['movieId'])
print(f"\n ID number of missing movie in 'movies' dataset: {len(missing_movies)}")


# %%
# Visualize the distribution of the number of ratings provided by users. 
# It shows how many users have contributed a specific number of ratings in the dataset.
import seaborn as sns
import matplotlib.pyplot as plt
user_ratings_count = rating['userId'].value_counts()  
sns.histplot(user_ratings_count, bins=50, kde=False, color= 'orchid')
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# %% [markdown]
# Most users have contributed with very few ratings, with a large spike at the lower end of the scale. This indicates a significant proportion of users are casual raters.
# 
# A small number of users have provided a disproportionately high number of ratings. These are likely highly active or dedicated users.
# 
# This distribution highlights the uneven engagement level among users.

# %%
#The graph displays the distribution of ratings in the dataset. 
import seaborn as sns

# Rating distribution
plt.figure(figsize=(8, 6))
sns.histplot(rating['rating'], bins=10, kde=True, color='crimson')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# The most frequent ratings are clustered around the higher end of the scale, especially near 4.0 and 5.0. This suggests that users tend to give favorable ratings to movies.
# 
# Ratings at the lower end (1.0 or 2.0) are less common, indicating that users are less likely to leave very negative reviews.
# 

# %%
# Extract release year from the title 
movie['year'] = movie['title'].str.extract(r'\((\d{4})\)').astype(float)

plt.figure(figsize=(10, 5))
sns.histplot(movie['year'].dropna(), bins=50, kde=False, color= 'plum')
plt.title('Number of Movies Released Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

# %% [markdown]
# The number of movies released increases significantly over time, particularly after the 1980s. This reflects advancements in the film industry, technology, and global accessibility to cinema.
# 
# The graph peaks in the early 2000s, indicating a boom in movie production during that period. This could be attributed to the development of digital technology.
# 
# Very few movies were produced before the 1920s.

# %%
# Top 20 most commonly used tags.
tag_counts = tag['tag'].value_counts().head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=tag_counts.values, y=tag_counts.index, palette='viridis')
plt.title('Top 20 Most Common Tags')
plt.xlabel('Count')
plt.ylabel('Tag')
plt.show()


# %% [markdown]
# Tags like "sci-fi," "based on a book," and "atmospheric" are the most commonly assigned, suggesting these themes are the most popular.
# 
# Many tags describe genres (e.g., "comedy," "action," "fantasy") or stylistic elements (e.g., "atmospheric," "surreal"), indicating the importance of these features in user tagging behavior.
# 

# %%
# Genome scores distribution
plt.figure(figsize=(8, 6))
sns.histplot(genome_scores['relevance'], bins=30, kde=True, color='hotpink')
plt.title('Distribution of Genome Relevance Scores')
plt.xlabel('Relevance')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# The relevance scores are heavily skewed toward lower values, with the majority clustered around 0.0–0.2. This indicates that most tags have low relevance for the movies in the dataset.
# 
# There is a gradual decline as the relevance scores increase, showing that only a small number of tags are highly relevant (closer to 1.0).
# 
# The skewed distribution implies that most tags may not strongly influence recommendations, while a few highly relevant tags could play a crucial role in tagging movies effectively and refining recommendations.

# %% [markdown]
# ### 1.2 Minhash Signatures 

# %% [markdown]
# For this point we grouped the dataset by `userId` and create a mapping where each user ID corresponds to the set of movies they have rated. This allows us to analyze user-specific preferences; then we calculated the Jaccard similarity to measure the similarity between two users' movie preferences. It calculates the number of common movies rated by both users to the total unique movies rated by either user.
# 
# Then, we used the method MinHash to estimate the Jaccard similarity. Instead of directly comparing sets, we compare compressed signature vectors generated through MinHashing. 
# 
# 

# %%
# Create a dictionary mapping each user ID to the set of movie IDs they have rated.
user_movies= rating.groupby('userId')['movieId'].apply(set).to_dict()

# %%
# Calculates the Jaccard similarity between two sets of movies.
def jaccard_similarity_exact(movies1, movies2):
    intersection = len(movies1 & movies2)  
    union = len(movies1 | movies2)         
    return intersection / union if union > 0 else 0

# %%
# Calculates the approximate Jaccard similarity between two MinHash signatures.

def jaccard_similarity_hashed(signature1, signature2):
    matches = sum(1 for x, y in zip(signature1, signature2) if x == y)
    return matches / len(signature1)


# %% [markdown]
# ### 1.2.1

# %% [markdown]
# In the next step, we define a linear hash functions that transform an input `x` using the formula `(a * x + b) % c`. 
# 
# The `generate_hash_function` creates a single hash function using a linear formula.
# 
# The `generate_hash_functions` generates a set of hash functions for MinHash using linear formulas.

# %%
import numpy as np

def generate_hash_function(a, b, c):

    def hash_function(x):
        return (a * x + b) % c  

def generate_hash_functions(num_hashes, max_movie_id, seed=None):

    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility.
    a = np.random.randint(1, max_movie_id, size=num_hashes)  # Random coefficients for a.
    b = np.random.randint(0, max_movie_id, size=num_hashes)  # Random coefficients for b.
    c = max_movie_id + 1 # Use max_movie_id + 1 as modulus (ensures unique outputs).
    return [generate_hash_function(a_, b_, c) for a_, b_ in zip(a, b)]  # Generate and return the hash functions.


# %% [markdown]
# In the followed steps, we will test three different MinHash functions: `2x`, `x^2`, `x // 2`, to determine which performs best in terms of accuracy and efficiency in finding similar users. We aim to identify the most accurate MinHash function for our recommendation system.

# %% [markdown]
# We introduce first a `2x` multiplier to the linear hash formula, which increases variability in the hash values. 
# 
# 

# %%
# Creates a hash function using a 2x multiplier in the formula.
def generate_hash_function_2x(a, b, c):

    def hash_function(x):
        return (a * 2 * x + b) % c  # Linear formula with 2x multiplier.
    return hash_function

# Generates a set of hash functions using the 2x multiplier
def generate_hash_functions_2x(num_hashes, max_movie_id, seed=None):

    if seed is not None:
        np.random.seed(seed)  
    a = np.random.randint(1, max_movie_id * 2, size=num_hashes)  
    b = np.random.randint(0, max_movie_id * 2, size=num_hashes)  
    c = max_movie_id * 2 + 1  
    return [generate_hash_function_2x(a_, b_, c) for a_, b_ in zip(a, b)]  



# %% [markdown]
# In the next code we use quadratic (x^2) method into the hash formula.

# %%
# Creates a hash function using a quadratic term in the formula.
def generate_hash_function_xe2(a, b, c):

    def hash_function(x):
        return (a * x / 2 + b * x) % c  # Combine linear and quadratic terms.
    return hash_function

# Generates a set of hash functions using quadratic terms.
def generate_hash_functions_xe2(num_hashes, max_movie_id, seed=None):

    if seed is not None:
        np.random.seed(seed) 
    a = np.random.randint(1, max_movie_id * 2, size=num_hashes)  
    b = np.random.randint(0, max_movie_id * 2, size=num_hashes)  
    c = max_movie_id * 2 + 1  # Use a modulus larger than max_movie_id.
    return [generate_hash_function_xe2(a_, b_, c) for a_, b_ in zip(a, b)]  


# %% [markdown]
# The next approach introduces division into the hash formula. By dividing values during hashing, we can test the impact of reduced output range on performance.

# %%
# Creates a hash function that incorporates division in the formula.
def generate_hash_function_div2(a, b, c):

    def hash_function(x):
        return (a * (x  // 2) + b) % c  # Introduce division into thehash formula
    return hash_function

# Generates a set of hash functions using division by 2 in the formula.
def generate_hash_functions_div2(num_hashes, max_movie_id, seed=None):
 
    if seed is not None:
        np.random.seed(seed)  
    a = np.random.randint(1, max_movie_id * 2, size=num_hashes)  
    b = np.random.randint(0, max_movie_id * 2, size=num_hashes)  
    c = max_movie_id * 2 + 1  # Use a modulus larger than max_movie_id.
    return [generate_hash_function_div2(a_, b_, c) for a_, b_ in zip(a, b)]  


# %% [markdown]
# ### 1.2.2

# %% [markdown]
# The `compute_signature` function generates MinHash signatures for a given set of users using specified hash functions. These signatures are compact representations of each user's movie preferences, that are created to test efficiently comparisons between users.

# %%
# Generates the MinHash signatures for a subset of users.
def compute_signature(sample_users, user_movies, hash_functions):
    #Empty dictionary to store the signature matrix.
    signature_matrix = {}
    
    for user in sample_users:
        # Retrieve the set of movies rated by the user.
        movies = user_movies.get(user, set())
        
        if not movies:
            # If no movies are rated, assign an empty signature.
            signature_matrix[user] = [float('inf')] * len(hash_functions)
            continue  
        
        signature = []
        
        # Iterate through each hash function.
        for h in hash_functions:
            # Compute the minimum hash value for all movies rated by the user.
            min_hash = min(h(movie) for movie in movies)
            # Append the minimum hash value to the user's signature.
            signature.append(min_hash)
        
        # Store the computed signature in the signature matrix.
        signature_matrix[user] = signature
    
    
    return signature_matrix



# %% [markdown]
# The `calculate_mse` function computes the Mean Squared Error (MSE) and it quantifies the accuracy of the MinHash-based similarity estimation.
# 
# 

# %%
def calculate_mse(user_movies, signature_matrix, sample_users):

    # Initialize MSE accumulator and pair count.
    mse = 0
    n_pairs = 0
    
    for i, user1 in enumerate(sample_users):
        for user2 in sample_users[i+1:]:
            # Calculate the exact Jaccard similarity.
            exact_jaccard = jaccard_similarity_exact(user_movies[user1], user_movies[user2])
            
            # Calculate the estimated Jaccard similarity using MinHash signatures.
            hashed_jaccard = jaccard_similarity_hashed(signature_matrix[user1], signature_matrix[user2])
            
            # Compute the squared error.
            mse += (exact_jaccard - hashed_jaccard) ** 2
            # Increment the count of user pairs.
            n_pairs += 1
    
    return mse / n_pairs if n_pairs > 0 else 0



# %% [markdown]
# We select a sample of 100 users and generate MinHash signatures for each user using the three different hash function we created before (2x multiplier, quadratic terms, and division by 2). 
# 

# %%
# Select the first 100 users for testing.
sample_users = list(user_movies.keys())[:100]
user_movies_sample = {user: user_movies[user] for user in sample_users}

# Generate hash functions using our three methods.
hash_functions_2x = generate_hash_functions_2x(num_hashes=10, max_movie_id=rating['movieId'].max(), seed=42)
hash_functions_xe2 = generate_hash_functions_xe2(num_hashes=10, max_movie_id=rating['movieId'].max(), seed=42)
hash_functions_div2 = generate_hash_functions_div2(num_hashes=10, max_movie_id=rating['movieId'].max(), seed=42)

# Compute MinHash signatures for the sample users.
signature_matrix_2x = compute_signature(sample_users, user_movies_sample, hash_functions_2x)
signature_matrix_xe2 = compute_signature(sample_users, user_movies_sample, hash_functions_xe2)
signature_matrix_div2 = compute_signature(sample_users, user_movies_sample, hash_functions_div2)

# Calculate the Mean Squared Error (MSE) for each type of hash function.
mse_2x = calculate_mse(user_movies, signature_matrix_2x, sample_users)
mse_xe2 = calculate_mse(user_movies, signature_matrix_xe2, sample_users)
mse_div2 = calculate_mse(user_movies, signature_matrix_div2, sample_users)


# %%
# Calculate the Mean Squared Error (MSE) for each hash function type.
print(f"MSE 2x: {mse_2x:.4f}")

print(f"MSE x^2: {mse_xe2:.4f}")

print(f"MSE x//2: {mse_div2:.4f}")


# %%
import matplotlib.pyplot as plt
# Create a bar plot to compare the Mean Squared Error (MSE) values for different hash functions.
mse_values = [mse_2x, mse_xe2, mse_div2]
labels = ['MSE 2x', 'MSE x^2', 'MSE x//2']

plt.figure(figsize=(8, 6))
plt.bar(labels, mse_values, color=['indigo', 'fuchsia', 'hotpink'], alpha=0.7)
plt.title('Comparison of MSE Values for Different Hash Functions')
plt.ylabel('MSE')
plt.xlabel('Hash Function')
plt.ylim(0, max(mse_values) + 0.001)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# %% [markdown]
# The bar plot provides a visual representation of the Mean Squared Error (MSE) for each hash function (`2x, x^2, x//2`).
# 
# 
# 1. **MSE `2x`:** the MSE value is relatively high, indicating that the MinHash similarity estimation is less accurate with this configuration.
# 2. **MSE `x^2`:** this has the lowest MSE, suggesting that this configuration provides the most accurate similarity estimates among the three tested.
# 3. **MSE `x//2`:** the MSE is slightly higher than the quadratic configuration but still better than the `2x` multiplier.
# 
# Among the tested hash functions, the quadratic (`x^2`) hash function is the best choice for minimizing errors in the similarity estimates.
# 
# 
# 

# %% [markdown]
# The next code generates two datasets:
# - `real_similarities`: Represents similarity between users.
# 
# - `estimated_similarities`: Represents the approximations derived from the MinHash algorithm.

# %%
# Calculate real and estimated similarities between user pairs.
real_similarities = []  
estimated_similarities = []  

# Iterate through all unique pairs of users in the sample.
for i, user1 in enumerate(sample_users):
    for user2 in sample_users[i+1:]:
        # Calculate the exact Jaccard similarity for the pair of users based on their movie sets.
        real_sim = jaccard_similarity_exact(user_movies[user1], user_movies[user2])
        real_similarities.append(real_sim)  

        # Calculate the estimated similarity using MinHash signatures.
        estimated_sim = jaccard_similarity_hashed(signature_matrix_xe2[user1], signature_matrix_xe2[user2])
        estimated_similarities.append(estimated_sim)  


# %% [markdown]
# ### 1.2.3

# %% [markdown]
# We tested the three hash function we created, across various similarity thresholds to determine the best-performing combinations.
# Thresholds were defined in the range [0, 1] with a step size of 0.1. 
# 
# At first we computed exact and estimated similarities between user pairs.
# Then we convert similarities into binary labels based on thresholds.
# 
# Then we compute precision, recall, and F1-score to see how well the estimated similarities align with the real ones at each threshold.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Define thresholds from 0 to 1 with step 0.1
thresholds = np.linspace(0, 1, 11)

# Dictionary storing the precomputed MinHash signatures for each hash function
hash_types = {
    "2x": signature_matrix_2x,
    "x^2": signature_matrix_xe2,
    "x//2": signature_matrix_div2
}

#  MSE values for each hash type
mse_values = {
    "2x": mse_2x,
    "x^2": mse_xe2,
    "x//2": mse_div2
}

results = []

for hash_name, signatures in hash_types.items():
    for threshold in thresholds:
        # Calculate real similarities using exact Jaccard similarity
        real_similarities = [
            jaccard_similarity_exact(user_movies_sample[user1], user_movies_sample[user2])
            for user1 in sample_users for user2 in sample_users if user1 != user2
        ]
        
        # Calculate estimated similarities using MinHash
        estimated_similarities = [
            jaccard_similarity_hashed(signatures[user1], signatures[user2])
            for user1 in sample_users for user2 in sample_users if user1 != user2
        ]

        # Create binary labels: 1 if similarity >= threshold, else 0
        real_labels = [1 if sim >= threshold else 0 for sim in real_similarities]
        estimated_labels = [1 if sim >= threshold else 0 for sim in estimated_similarities]

        # Compute Precision, Recall, and F1-Score
        precision = precision_score(real_labels, estimated_labels, zero_division=0)
        recall = recall_score(real_labels, estimated_labels, zero_division=0)
        f1 = f1_score(real_labels, estimated_labels, zero_division=0)

        mse = mse_values[hash_name]

        results.append((hash_name, threshold, precision, recall, f1, mse))


df_results = pd.DataFrame(results, columns=['Hash Type', 'Threshold', 'Precision', 'Recall', 'F1-Score', 'MSE'])

print(df_results)


# %% [markdown]
# Results: 
# 
# `x^2` (Quadratic) achieved the best F1-scores across all thresholds.
# It has also the lowest Mean Squared Error (MSE) so it is the most accurate configuration for estimating user similarities.
# 
# `x//2`performed well at mid-range thresholds but slightly worst compared to `x^2` at lower thresholds.
# MSE values were slightly higher than `x^2`, indicating less 
# 
# `2x`had the highest MSE and the lowest F1-scores across most thresholds.
# Precision improved at higher thresholds, but the overall performance was less effective than other configurations.
# 
# 
# The `x^2` configuration performed the best overall, with high F1-scores and the lowest MSE values. It is the most effective configuration for MinHash-based similarity estimation.
# 
# Mid-range thresholds (0.4–0.7) provide the best balance between precision and recall, maximizing the F1-score for all hash functions. 
# 
# Lower thresholds ( 0.0–0.2) favor recall but penalize precision.
# 
#  Higher thresholds ( ≥ 0.5) result in high precision but very low recall.
# 

# %%
# Compute the MinHash signature matrix for the sampled users using the x^2 hash functions.

signature_matrix_xe2_example = compute_signature(user_movies_sample, user_movies_sample, hash_functions_xe2)


# %%
# Generates the MinHash signature for each user.
def compute_signature(user_movies, hash_functions):

    signature_matrix = {}
    
    # Iterate through each user and their set of movies.
    for user, movies in user_movies.items():
        signature = []  
        # Apply each hash function to the user's movies to compute the minimum hash value.
        for h in hash_functions_xe2:  # Use x^2 hash function.
            min_hash = min(h(movie) for movie in movies)  # Compute the minimum hash value for this user and function.
            signature.append(min_hash)  
        
        signature_matrix[user] = signature
    
    return signature_matrix  

# Example usage
num_hashes = 100  
hash_functions = generate_hash_functions(num_hashes, max_movie_id=100000)  # Generate hash functions with max_movie_id=100,000.
signature_matrix = compute_signature(user_movies, hash_functions)  # Compute the signature matrix for all users.


# %%
# Computes the MinHash signature matrix for all users in the user_movies dictionary.
signature_matrix = compute_signature(user_movies, hash_functions)

# %% [markdown]
# ### 1.3 Locality-Sensitive Hashing (LSH)

# %% [markdown]
# ### 1.3.1

# %% [markdown]
# This function divides the MinHash signatures into multiple bands and hashes to create buckets. Users with similar signatures in a band are grouped in the same bucket.
# 
# Then we computed the Jaccard similarity between two users using their MinHash signatures. 

# %%
# Create buckets for LSH by dividing the signature matrix into bands.
def lsh_buckets(signature_matrix, num_bands):

    from collections import defaultdict

    buckets = defaultdict(list) # Initialize buckets to store user groups for each band.
    rows_per_band = len(next(iter(signature_matrix.values()))) // num_bands
    if len(next(iter(signature_matrix.values()))) % num_bands != 0:
        raise ValueError("The length of signatures must be divisible by the number of bands.")


    for user, signature in signature_matrix.items():
        for i in range(num_bands):
            # Extract rows for the current band
            start_index = i * rows_per_band
            end_index = start_index + rows_per_band
            band = tuple(signature[start_index:end_index])
            # Hash the band to create a bucket key
            band_hash = hash(band)
            buckets[band_hash].append(user)
    
    return buckets

# Example:
num_bands = 2 # Divide the signature into 5 bands
buckets = lsh_buckets(signature_matrix, num_bands)


for band_hash, users in list(buckets.items())[:10]:
    print(f"Band Hash: {band_hash}, Users in Bucket: {users}")


# %%
# Calculates the Jaccard similarity between two users using their signatures from the selected buckets.
def jaccard_similarity(user1, user2, buckets):

    # Retrieve the MinHash signature for user1 from the signature matrix.
    user1_signature = signature_matrix[user1]
    
    # Retrieve the MinHash signature for user2 from the signature matrix.
    user2_signature = signature_matrix[user2]
    
    # Calculate common elements between the two sets.
    intersection = len(set(user1_signature) & set(user2_signature))
    
    # Calculate unique elements across the two sets.
    union = len(set(user1_signature) | set(user2_signature))
    
    # Return the Jaccard similarity 
    return intersection / union

# Example:
# Compute Jaccard similarity between user 5 and user 80396 using the specified buckets.
print(jaccard_similarity(5, 80396, buckets))


# %% [markdown]
# In the next step we check if all users in the same bucket have consistent signatures for a given band. If there is more than one unique signature in the bucket, then it's inconsistent.
# 

# %%

# Checks the consistency of signatures in the selected buckets.
def check_signature_consistency(buckets, signatures, band_index, rows_per_band):

    start = band_index * rows_per_band
    end = start + rows_per_band

    for bucket_id, users in buckets.items():
        # Extract the signatures for the users in the current bucket for the specified band.
        bucket_signatures = [signatures[user][start:end] for user in users]
        
        # Check if there are inconsistencies in the bucket by comparing the signatures.
        if len(set(map(tuple, bucket_signatures))) > 1:
            print(f"Inconsistency found in Bucket {bucket_id}")  
        else:
            print(f"Bucket {bucket_id} is consistent") 



# %% [markdown]
# We iterate through all bands and verify the consistency of buckets for each band. This help us to ensure that users in the same bucket have similar signatures within each band. We need this process to debug the LSH process.

# %%
# Verifies the consistency of all buckets across all bands.
def verify_all_buckets(buckets, signature_matrix, num_bands, rows_per_band):

    for band_index in range(num_bands):
        # Print a progress message indicating the current band being verified.
        print(f"\nVerifying band {band_index + 1}/{num_bands}:")
        
        # Check the consistency of buckets for the current band.
        check_signature_consistency(buckets, signature_matrix, band_index, rows_per_band)

rows_per_band = len(next(iter(signature_matrix.values()))) // num_bands

verify_all_buckets(buckets, signature_matrix, num_bands, rows_per_band)



# %% [markdown]
# The `analyze_buckets` function evaluates the distribution of users across all buckets calculating the size of each bucket, 
# computing  the average bucket size and identifying the largest and smallest buckets counting users.
# 

# %%
# Analyzes the distribution of users across buckets
def analyze_buckets(buckets):

    # Calculate number of users of each bucket.
    bucket_sizes = [len(users) for users in buckets.values()]
    
    # Print the average number of users per bucket.
    print(f"Average number of users per bucket: {sum(bucket_sizes) / len(bucket_sizes)}")
    
    # Print the size of the largest bucket.
    print(f"Largest bucket contains {max(bucket_sizes)} users")
    
    # Print the size of the smallest bucket.
    print(f"Smallest bucket contains {min(bucket_sizes)} users")


# %%
analyze_buckets(buckets)

# %%
#  Identifies inconsistent buckets based on size criteria.
def find_inconsistent_buckets(buckets, min_users=2, max_users=100):
 
    # Initialize a set to store the hashes of inconsistent buckets.
    inconsistent_buckets = set()
    
    for band_hash, users in buckets.items():
        # Check if the number of users is outside the specified range.
        if len(users) < min_users or len(users) > max_users:
            inconsistent_buckets.add(band_hash)  
    return inconsistent_buckets

# Example: 
# Identify inconsistent buckets based on a minimum of 1 user and a maximum of 1000 users.
inconsistent_buckets = find_inconsistent_buckets(buckets, min_users=1, max_users=1000)


print(len(inconsistent_buckets))



# %% [markdown]
# ### 1.3.2

# %% [markdown]
# The function `find_similar_users` identifies all users who share at least one bucket with the given user. 
# 

# %%
def find_similar_users(user_id, buckets):

    similar_users = set()
    #Find the bucket with the given user
    for bucket_users in buckets.values():
        if user_id in bucket_users: 
            similar_users.update(bucket_users)
    similar_users.discard(user_id) 
    return similar_users


# %% [markdown]
# The next function adjusts the number of bands in LSH to ensure buckets can be created. If initial parameters fail to generate meaningful buckets, it reduces the number of bands and retries, preventing empty or inconsistent bucket assignments.
# 
# 

# %%
# Adjusts LSH parameters to generate buckets if no similar users are found initially.
def adjust_lsh_parameters(signature_matrix, initial_num_bands, max_attempts=5):

    num_bands = initial_num_bands
    for _ in range(max_attempts):
        try:
            buckets = lsh_buckets(signature_matrix, num_bands)
            return buckets 
        except ValueError:
            num_bands -= 1
            if num_bands < 1:
                raise ValueError("Cannot adjust LSH parameters further.")

    return {}

# %% [markdown]
# This function computes and ranks the similarity scores between the target user and all similar users. It uses the Jaccard similarity metric based on their MinHash signatures.
# 
# 

# %%
#  Ranks similar users based on their similarity to the target user.
def rank_similar_users(user_id, similar_users, signature_matrix):
 
    ranked_users = []
    user_signature = signature_matrix[user_id]

    for similar_user in similar_users:
        similar_user_signature = signature_matrix[similar_user] 
        similarity_score = jaccard_similarity_hashed(user_signature, similar_user_signature)  
        ranked_users.append((similar_user, similarity_score))  
    ranked_users.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_users


# %% [markdown]
# The next step combines all previous steps to identify and rank the top 2 most similar users for a given target user. If initially the result are not representative, it adjusts the LSH parameters to refine the bucket assignments.
# 
# 

# %%
#  Finds the top 2 most similar users for a given user based on LSH buckets.
def find_top_similar_users(user_id, signature_matrix, buckets, initial_num_bands=10):

    similar_users = find_similar_users(user_id, buckets)
    if len(similar_users) <= 1:
        print("Adjusting LSH parameters...")
        buckets = adjust_lsh_parameters(signature_matrix, initial_num_bands)  
        similar_users = find_similar_users(user_id, buckets)  

    ranked_users = rank_similar_users(user_id, similar_users, signature_matrix)
    return ranked_users[:2]



# %% [markdown]
# We tried to do an example: the system finds the top 2 most similar users for the target user (ID 254). 
# It first identifies users sharing at least one bucket and then ranks them based on similarity scores. If necessary, it adjusts the LSH parameters to ensure meaningful results.

# %%

user_id = 254
top_similar_users = find_top_similar_users(user_id, signature_matrix, buckets, initial_num_bands=10)

print(f"Top 2 similar users for {user_id}: {top_similar_users}")


# %% [markdown]
# 
# The system initially failed to find enough similar users for the target user (ID 254) using the initial LSH parameters.
# So the LSH parameters  were adjusted to refine bucket assignments.
# 
# The two most similar users for the target user are identified as:
# - User 23880 with a similarity score of 0.5.
# 
# - User 78715 with a similarity score of 0.5.
# 
# These scores indicate that both users share 50% similarity (Jaccard similarity) with the target user based on their MinHash signatures.
# 
# 

# %% [markdown]
# ### 1.3.3

# %% [markdown]
# The first line of code creates a dictionary `movie_titles` where each key is a movieId and the value is the corresponding movie title. This mapping is used to convert `movieId` values in movie titles.

# %%
# Creates a dictionary mapping movie IDs to their corresponding titles.
movie_titles = dict(zip(movie["movieId"], movie["title"]))

# %% [markdown]
# The next function recommends movies for a user: if the top two similar users have rated the same movies, it recommends those movies based on their average ratings.
# 
# If there are no common movies, it recommends the highest-rated movies of the most similar user. 

# %%
# Recommends movies for a given user based on similar users' ratings.
def recommend_movies(user_id, user_ratings, num_recommendations=5):

    # Identify the two most similar users using their MinHash signatures and buckets.
    top_similar_users = find_top_similar_users(user_id, signature_matrix, buckets, initial_num_bands=10)
    similar_user_1, similar_user_2 = top_similar_users[0][0], top_similar_users[1][0]
    
    # Get the ratings of both similar users
    ratings_1 = user_ratings.get(similar_user_1, {})
    ratings_2 = user_ratings.get(similar_user_2, {})
    
    # Find commonly rated movies
    common_movies = set(ratings_1.keys()).intersection(set(ratings_2.keys()))
    
    if common_movies:
        # Recommend movies based on average rating
        recommendations = [
            (movie, (ratings_1[movie] + ratings_2[movie]) / 2) for movie in common_movies
        ]
        # Sort movies by their average rating in descending order.
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    else:
        # If no common movies, recommend top-rated movies of the most similar user
        top_rated_movies = sorted(ratings_1.items(), key=lambda x: x[1], reverse=True)
        return top_rated_movies[:num_recommendations]


# %%
# Recommends movies for a given user based on their most similar users.
def query_recommendation(user_id, user_ratings, signature_matrix, buckets, num_recommendations=5):

    # Identify the top 2 most similar users
    top_similar_users = find_top_similar_users(user_id, signature_matrix, buckets, initial_num_bands=10)
    if len(top_similar_users) < 2:
        raise ValueError("Not enough similar users found for recommendations.")

    similar_user_1, similar_user_2 = top_similar_users[0][0], top_similar_users[1][0]

    # Fetch the ratings of the two most similar users
    ratings_1 = user_ratings.get(similar_user_1, {})
    ratings_2 = user_ratings.get(similar_user_2, {})

    # Find common movies rated by both similar users
    common_movies = set(ratings_1.keys()).intersection(set(ratings_2.keys()))
    recommendations = []

    if common_movies:
        # Calculate average ratings for common movies
        recommendations = [
            (movie, (ratings_1[movie] + ratings_2[movie]) / 2) for movie in common_movies
        ]

    # Sort by average rating in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Add additional movies from the most similar user if needed.
    if len(recommendations) < num_recommendations:
        additional_movies = sorted(ratings_1.items(), key=lambda x: x[1], reverse=True)
        for movie, score in additional_movies:
            if movie not in [rec[0] for rec in recommendations]:
                recommendations.append((movie, score))
                if len(recommendations) >= num_recommendations:
                    break

    # Ensure the list has enough recommendations
    if len(recommendations) < num_recommendations:
        # Add movies from the second most similar user
        additional_movies = sorted(ratings_2.items(), key=lambda x: x[1], reverse=True)
        for movie, score in additional_movies:
            if movie not in [rec[0] for rec in recommendations]:
                recommendations.append((movie, score))
                if len(recommendations) >= num_recommendations:
                    break

    
    return recommendations[:num_recommendations]


# %%
# Recommends movies for a given user with titles instead of IDs.
def query_recommendation_with_titles(user_id, user_ratings, signature_matrix, buckets, movie_titles, num_recommendations=5):
 
    # Get movie recommendations with movie IDs.
    recommendations = query_recommendation(user_id, user_ratings, signature_matrix, buckets, num_recommendations)

    # Replace movie IDs with titles.
    recommendations_with_titles = [
        #  If a title is missing, it labels the movie as "Unknown Movie (ID: [movie_id])".
        (movie_titles.get(movie_id, f"Unknown Movie (ID: {movie_id})"), score)
        for movie_id, score in recommendations
    ]

    return recommendations_with_titles



# %%
# Generate the `user_ratings` dictionary from the `rating` DataFrame.

user_ratings = rating.groupby("userId").apply(
    lambda group: dict(zip(group["movieId"], group["rating"]))
).to_dict()


# %% [markdown]
# ### 1.3.4

# %% [markdown]
# In the next step we attempt to find the top 5 recommended movies for user 254 by first identifying the two most similar users using the LSH technique.
# If the initial bucket configuration fails it automatically adjusts the LSH parameters to improve clustering.

# %%
# Define the target user and number of recommendations
Id = 254
num_recommendations = 5

# Get movie recommendations with titles
raccomandation = query_recommendation_with_titles(
    user_id, user_ratings, signature_matrix, buckets, movie_titles, num_recommendations=5
)

# Print the top 5 recommended movies
for i in range(0, num_recommendations):
    print(f"Top 5 recommended movies for user {Id}: {raccomandation[i]}")


# %% [markdown]
# 
# The message "Adjusting LSH parameters..." indicates that the initial configuration was not sufficient or meaningful recommendations, so the LSH parameters were adjusted.
# 
# The system successfully generated five recommendations for user 254. All movies have a perfect score of 5.00, indicating that they were highly rated by the similar users.
# 
# These recommendations are highly relevant to user 254 because they are based on the preferences of users identified as most similar. This ensures that the suggested movies align with the user's tastes.
# 

# %% [markdown]
# ## **Gouping Movies Together!**

# %% [markdown]
# ### 2.1 Feature Engineering

# %%
import pandas as pd

# %%
# Read all datasets
genome_scores = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/genome_scores.csv")
genome_tags = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/genome_tags.csv")
link = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/link.csv")
movie = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/movie.csv")
rating = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/rating.csv")
tag = pd.read_csv("C:/Users/hp/Desktop/movielens-20m-dataset/tag.csv")

# %%
movie = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/movie.csv')
rating = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/rating.csv')
genome_scores = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/genome_scores.csv') 
tag = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/tag.csv')
genome_tags = pd.read_csv('C:/Users/EMILIO/Documents/università/ADM/HMW 4/genome_tags.csv')

# %%
movies = movie.copy()

# %% [markdown]
# This script creates a set of features to represent movies more effectively for clustering analysis. The steps include:
# 1. Extracting unique genres and representing them as binary features.
# 2. Calculating the average ratings for each movie.
# 3. Determining the most relevant genome tag for each movie.
# 4. Finding the most common user tag for each movie.
# 5. Adding additional features like the number of ratings, rating variance, genre count, release year, and the time between the first and last rating for each movie.
# 
# 

# %%
# 1. Extract unique genres and create binary columns for each genre
unique_genres = set("|".join(movies["genres"]).split("|"))
for genre in unique_genres:
    # Create a binary column for each genre (1 if the movie belongs to that genre, 0 otherwise)
    movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x else 0)

# 2. Calculate the average ratings for each movie
ratings_avg = rating.groupby("movieId")["rating"].mean().rename("ratings_avg")
# Merge the average ratings with the movies dataset
movies = movies.merge(ratings_avg, on="movieId", how="left")

# 3. Find the most relevant genome tag for each movie
genome_scores = genome_scores.sort_values(["movieId", "relevance"], ascending=[True, False])
# Get the most relevant tag for each movie
relevant_tags = genome_scores.groupby("movieId").first().reset_index()
relevant_tags = relevant_tags.rename(columns={"relevance": "relevant_genome_tag"})

# Merge the relevant genome tags with the movies dataset
movies = movies.merge(relevant_tags[["movieId", "tagId", "relevant_genome_tag"]], on="movieId", how="left")
# Replace missing tagId values with -1
movies["tagId"] = movies["tagId"].fillna(-1).astype(int)
# Add the tag descriptions and drop the tagId column
movies = movies.merge(genome_tags, on="tagId", how="left")
movies = movies.drop(columns=["tagId"])

# 4. Find the most common user tag for each movie
common_tags = tag.groupby("movieId")["tag"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
# Map the most common tags to the movies dataset
movies["common_user_tag"] = movies["movieId"].map(common_tags)

# 5. Calculate the number of ratings for each movie
num_ratings = rating.groupby("movieId")["rating"].count().rename("number_of_ratings")
# Merge the number of ratings with the movies dataset
movies = movies.merge(num_ratings, on="movieId", how="left")

# 6. Calculate the variance of ratings for each movie
rating_variance = rating.groupby("movieId")["rating"].var().rename("rating_variance")
# Merge the rating variance with the movies dataset
movies = movies.merge(rating_variance, on="movieId", how="left")

# 7. Count the number of genres for each movie
movies["genre_count"] = movies["genres"].apply(lambda x: len(x.split("|")))

# Extract the release year from the movie title and remove it from the title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=True)  # Extract the year in parentheses
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)  # Remove the year from the title

# Drop the original genres column 
movies = movies.drop(columns=['genres'])

# 8. Calculate the time elapsed between the first and last rating for each movie
rating['timestamp'] = pd.to_datetime(rating['timestamp'])
# Group by movieId and calculate the min and max timestamps
time_diff = rating.groupby('movieId')['timestamp'].agg(['min', 'max'])
time_diff['time_elapsed'] = (time_diff['max'] - time_diff['min']).dt.days

# Merge the time elapsed with the movies dataset
movies = movies.merge(time_diff['time_elapsed'], on='movieId', how='left')


print("The movies dataset has been updated with new features for clustering:")
print(movies.head())


# %%
movies.head()

# %%
# Check for missing values
movies.isna().sum()

# %%
# Plot a heatmap to show null values
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(movies.isnull(), cbar=False)
plt.show()

# %%
# Fill missing values with appropriate replacements
movies['tag'].fillna('Unknown', inplace=True)
movies['common_user_tag'].fillna('Unknown', inplace=True)
movies['ratings_avg'].fillna(movies['ratings_avg'].mean(), inplace=True)
movies['rating_variance'].fillna(0, inplace=True)
movies['number_of_ratings'].fillna(0 , inplace=True)
movies['relevant_genome_tag'].fillna(movies['relevant_genome_tag'].mean(), inplace=True)
movies.dropna(subset=['year'], inplace=True)
movies['time_elapsed'].fillna(0, inplace=True)

# %%
# Recheck the dataset for missing values
movies.isna().sum()

# %% [markdown]
# ### 2.2. Choose your features (variables)!

# %% [markdown]
# ### 2.2.1 - 2.2.2

# %% [markdown]
# 
# Normalization is a very important step in data preprocessing, in particular for clustering algorithms like KMeans; in fact these algorithms rely on distance-based metrics such as Euclidean distance. 
# Features with different scales can influence the clustering process; without normalization, features with larger ranges overshadow those with smaller ones, leading to biased and distort results. Normalization ensures that all features contribute equally by transforming them to a common scale.
# 
# Since normalizing data has a clear benefit, we proceed to normalize the numeric variables in the dataset, using the MinMaxScaler from scikit-learn; we scale the data to a range between 0 and 1, ensuring that all features have a comparable magnitude. 
# 

# %%
# Remove non-numerical and unnecessary columns
movies_num = movies.drop(columns=['movieId', 'title','tag', 'common_user_tag'])
movies_num

# %%
from sklearn.preprocessing import MinMaxScaler
# Transform the numerical dataset to scale features between 0 and 1.
scaler = MinMaxScaler()
movies_normalized = scaler.fit_transform(movies_num)
movies_normalized = pd.DataFrame(movies_normalized, columns=movies_num.columns)
movies_normalized

# %%
# Reset indices of non-numerical columns and concatenate with normalized features
movies_reset = movies[['tag', 'common_user_tag']].reset_index(drop=True)
movies_normalized_reset = movies_normalized.reset_index(drop=True)
movies_cluster = pd.concat([movies_reset, movies_normalized], axis=1)

# %%
movies_cluster

# %% [markdown]
# ### 2.2.3-2.2.4 

# %% [markdown]
# Dimensionality reduction is a process of reducing the number of features in a dataset while retaining as much information as possible. This can improve computational efficiency, reduce noise, and improve model interpretability.
# 
# For this task we used Principal Component Analysis (PCA) that is a statistical technique that transforms the dataset into a set of orthogonal components, known as principal components, which capture the maximum variance in the data. The first component captures the most variance, the second captures the next most, and so on. 
# 
# For this point we analyzed three different scenarios to determine whether dimensionality reduction was convinient or not. We conducted three test: 
# 
# - **Test 1**: we applied PCA to all the normalized numerical values. 
# 
# - **Test 2**: we modified cathegorical variables such as `tag` or `common_user_tag` using `Word2Vec` and then we applied PCA to all variables (numerical and cathegorical).
# 
# - **Test 3**: we didn't apply PCA.

# %% [markdown]
# ### **Test 1**

# %%
# Remove non-numerical columns ('tag' and 'common_user_tag').
movies_numeric = movies_cluster.drop(columns=['tag', 'common_user_tag'])
movies_numeric

# %%
# Perform PCA on the numerical dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(movies_numeric)

# Calculate the cumulative explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Principal Components")
plt.show()


# %% [markdown]
# The first 5–10 components capture most of the variance.
# 
# After ~15 components, the curve flattens, meaning additional components add little variance.
# 
# Around 5–10 components balance variance retention and dimensionality reduction. Selecting these components effectively reduces dimensionality while preserving most of the dataset's information.

# %% [markdown]
# Initialize PCA with an optimal number of components: based on the cumulative explained variance graph, we select 5 components to retain most of the variance

# %%
pca = PCA(n_components=5)

pca.fit(movies_numeric)
data_pca = pca.transform(movies_numeric)
reduced_dataset = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
print("Reduced dataset with principal components:")
print(reduced_dataset.head())

# %%
# Save the reduced dataset for later use as Dataset_1
dataset_1 = reduced_dataset.copy()

# %% [markdown]
# _____________________________

# %% [markdown]
# ### **Test 2**

# %%
# Split numerical and textual data.
movies_tags = movies[['tag', 'common_user_tag']]
movies_numeric = movies.drop(columns=['tag', 'common_user_tag', 'title', 'movieId'])

# %%
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Tokenize textual columns
movies_tags['tags_tokenized'] = movies_tags['tag'].str.split()
movies_tags['common_tags_tokenized'] = movies_tags['common_user_tag'].str.split()

#  Combine tokenized tags and common tags into a single list for each movie
movies_tags['all_tags_tokenized'] = movies_tags['tags_tokenized'] + movies_tags['common_tags_tokenized']

# Prepare a corpus (list of lists of words) for Word2Vec training
corpus = movies_tags['all_tags_tokenized'].tolist()
print(corpus)
# Output: [['action', 'adventure', 'hero', 'heroic', 'epic'], ...]


# %%
# Train a Word2Vec model
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=corpus,    
    vector_size=50,      
    window=5,            
    min_count=1,         
    workers=4,           
    sg=0                 
)

model.save("word2vec_model")

# %%
import numpy as np
#  Calculate average word vectors for each movie
# Compute the average vector for a list of tokens
def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

# Apply the function to tags and common tags
movies_tags['vector_tags'] = movies_tags['tags_tokenized'].apply(lambda x: sentence_vector(x, model))
movies_tags['vector_common_tags'] = movies_tags['common_tags_tokenized'].apply(lambda x: sentence_vector(x, model))


# %%
# Expand vectors into individual columns
vector_tags_expanded = pd.DataFrame(
    movies_tags['vector_tags'].tolist(),
    columns=[f'vector_tag_{i}' for i in range(model.vector_size)]
)

vector_common_expanded = pd.DataFrame(
    movies_tags['vector_common_tags'].tolist(),
    columns=[f'vector_common_tag_{i}' for i in range(model.vector_size)]
)
vector_tags_expanded.index = movies_numeric.index
vector_common_expanded.index = movies_numeric.index
movies_vectors = pd.concat([movies_numeric, vector_tags_expanded, vector_common_expanded], axis=1)
movies_vectors


# %%
# Reduce dimensions to 2 components for clustering
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(movies_vectors)

data_normalized = pd.DataFrame(normalized_data, columns=movies_vectors.columns)
data_normalized

# %%
# Perform PCA and visualize explained variance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA to the normalized dataset
pca = PCA()
pca.fit(data_normalized)

# Calculate cumulative explained variance
explained_variance_ratio = pca.explained_variance_ratio_.cumsum()

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Principal Components")
plt.show()

# %% [markdown]
# The first 20–30 components capture most of the variance.
# 
# After ~40 components, additional ones contribute minimally.
# 
# Almost 100% of variance is explained with all components.
# 
# Selecting ~20–30 components balances variance retention and efficiency. Components beyond 40 mostly capture noise or redundant information.

# %%
# Reduce dimensions to 2 components for clustering
pca = PCA(n_components=2)
pca.fit(data_normalized)
data_pca = pca.transform(data_normalized)
dataset_2 = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
dataset_2

# %% [markdown]
# _______________________

# %% [markdown]
# ### **Test 3**

# %%
# Here we do not apply PCA
# Dataset 3 preserves all original numerical features introducing additional transformations.
dataset_3 = movies_normalized.copy()
dataset_3

# %% [markdown]
# ### 2.3 Clustering 

# %% [markdown]
# For this step, we followed the approach outlined below:
# 
# - We implemented K-Means from scratch using the MapReduce;
# 
# - We implemented K-Means++ for improved centroid initialization.
# 
# - We used three methods: Elbow Method, Silhouette Score, and k-Distance Graph, to determine the optimal number of clusters for the three datasets created in the previous exercise.
# 
# - For each dataset, we compared the performance of the clustering methods (K-Means, K-Means++, and DBSCAN which was suggested by the LLM tool (exercise 2.3.4)) to identify the most effective approach for clustering.

# %%
import numpy as np
import random
from collections import defaultdict

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def initialize_centroids(data, num_clusters):
    return random.sample(list(data), num_clusters)
def mapper(data, centroids):
    mapped_data = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        mapped_data.append((cluster, point))
    return mapped_data

def reducer(mapped_data):
    cluster_points = defaultdict(list)
    for cluster, point in mapped_data:
        cluster_points[cluster].append(point)
    new_centroids = []
    for cluster, points in cluster_points.items():
        new_centroids.append(np.mean(points, axis=0))
    return new_centroids

def kmeans(data, num_clusters, max_iterations=100):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    np.random.seed(42)  
    centroids = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
    
    for iteration in range(max_iterations):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(num_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# %%
import numpy as np

# Function to initialize centroids using K-means++ algorithm
def initialize_centroids_kmeans_plus_plus(data, k, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)
    data = np.array(data)
    centroids = []
    centroids.append(data[np.random.choice(data.shape[0])])

    for _ in range(1, k):
        distances = np.min([np.linalg.norm(data - c, axis=1)**2 for c in centroids], axis=0)
        probabilities = distances / distances.sum()
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()
        for idx, prob in enumerate(cumulative_probs):
            if r < prob:
                centroids.append(data[idx])
                break
    
    return np.array(centroids)


# %%
import numpy as np

# K-means clustering with K-means++ initialization
def kmeans_with_kmeans_plus_plus(data, k, max_iterations=300, random_state=None):
    """
    Perform K-means clustering with K-means++ initialization.
    """
    if random_state is not None:
        np.random.seed(random_state)
    data = np.array(data)
    centroids = initialize_centroids_kmeans_plus_plus(data, k, random_state)
    
    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels


# %% [markdown]
# _________________

# %% [markdown]
# ### **Elbow Method, Silhouette Score and K-distance for Dataset_1**

# %%
from scipy.spatial.distance import cdist

wcss = []
for k in range(1, 11):  # Try from 1 to 10 clusters
    centroids, labels = kmeans(dataset_1, num_clusters=k)
    distances = cdist(dataset_1, centroids, metric='euclidean')
    wcss.append(sum(np.min(distances, axis=1)**2))

# Plot the Elbow Method
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o') 
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')  
plt.title('Elbow Method')  


# %% [markdown]
# The WCSS decreases rapidly around 4–6 clusters, after which the rate of decrease significantly.
# The elbow point suggests the optimal number of clusters is 4–6.

# %%
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_scores = []

# Calculate Silhouette Score for different numbers of clusters
for k in range(2, 11):
    centroids, labels = kmeans(dataset_1, num_clusters=k)  
    score = silhouette_score(dataset_1, labels)  
    silhouette_scores.append(score)

# Plot the results
plt.plot(range(2, 11), silhouette_scores, marker='o')  
plt.xlabel('Number of Clusters')  
plt.ylabel('Silhouette Score')  
plt.title('Silhouette Method')  
plt.show()



# %% [markdown]
# The silhouette score peaks at 9 clusters, indicating that clustering with 9 clusters achieves the best-defined groups.

# %%
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

nearest_neighbors = NearestNeighbors(n_neighbors=5) 
nearest_neighbors.fit(dataset_1) 
distances, indices = nearest_neighbors.kneighbors(dataset_1) 
distances = np.sort(distances[:, -1])  

plt.plot(distances)  
plt.title("k-Distance Graph")  
plt.xlabel("Points sorted by distance")  
plt.ylabel("k-distance")  
plt.show()


# %% [markdown]
# The steep increase in distances near the end indicates the optimal epsilon value for DBSCAN clustering.
# The "knee" in the graph indicates the distance threshold where points transition from being in dense clusters to being outliers or noise.
# This point is critical for selecting the epsilon value. 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# Comparing outputs

# %%
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

num_clusters = 6
centroids_kmeans, labels_kmeans = kmeans(dataset_1, num_clusters, max_iterations=300)
silhouette_kmeans = silhouette_score(dataset_1, labels_kmeans) if len(np.unique(labels_kmeans)) > 1 else -1
centroids_kmeans_plus, labels_kmeans_plus = kmeans_with_kmeans_plus_plus(dataset_1, num_clusters, max_iterations=300)
silhouette_kmeans_plus = silhouette_score(dataset_1, labels_kmeans_plus) if len(np.unique(labels_kmeans_plus)) > 1 else -1
dbscan = DBSCAN(eps=0.08, min_samples=5)  
labels_dbscan = dbscan.fit_predict(dataset_1)
silhouette_dbscan = silhouette_score(dataset_1, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else -1

print(f"Silhouette KMeans (random initialization): {silhouette_kmeans}")
print(f"Silhouette KMeans++: {silhouette_kmeans_plus}")
print(f"Silhouette DBSCAN: {silhouette_dbscan}")


# %% [markdown]
# ________________

# %% [markdown]
# ### **Elbow Method, Silhouette Score and K-distance for Dataset_2**

# %%
from scipy.spatial.distance import cdist

wcss = []
for k in range(1, 11): 
    centroids, labels = kmeans(dataset_2, num_clusters=k) 
    distances = cdist(dataset_2, centroids, metric='euclidean')  
    wcss.append(sum(np.min(distances, axis=1)**2)) 

# Plot the Elbow Method
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o') 
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')  
plt.title('Elbow Method') 
plt.show()


# %% [markdown]
# The "elbow" is observed at 3 clusters, suggesting it as the optimal number for KMeans.

# %%
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_scores = []

# Calculate the Silhouette Score for different numbers of clusters
for k in range(2, 11):  
    centroids, labels = kmeans(dataset_2, num_clusters=k)  
    score = silhouette_score(dataset_2, labels)  
    silhouette_scores.append(score)

# Plot the Silhouette Scores
plt.plot(range(2, 11), silhouette_scores, marker='o')  
plt.xlabel('Number of Clusters')  
plt.ylabel('Silhouette Score')  
plt.title('Silhouette Method')  
plt.show()


# %% [markdown]
# The highest silhouette score is at 3 clusters, confirming 3 as the most well-defined cluster number.

# %%
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# Fit the nearest neighbors model to the dataset
nearest_neighbors = NearestNeighbors(n_neighbors=5)  
nearest_neighbors.fit(dataset_2)
distances, indices = nearest_neighbors.kneighbors(dataset_2)  

# Sort distances for the k-distance graph
distances = np.sort(distances[:, -1])  
plt.plot(distances)  
plt.title("k-Distance Graph")  
plt.xlabel("Points sorted by distance")  
plt.ylabel("k-distance")  
plt.show()


# %% [markdown]
# The "knee" in the curve indicates the optimal eps value for DBSCAN clustering, which separates dense clusters from noise.

# %%
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

num_clusters = 3
centroids_kmeans, labels_kmeans = kmeans(dataset_2, num_clusters, max_iterations=300)
silhouette_kmeans = silhouette_score(dataset_2, labels_kmeans) if len(np.unique(labels_kmeans)) > 1 else -1
centroids_kmeans_plus, labels_kmeans_plus = kmeans_with_kmeans_plus_plus(dataset_2, num_clusters, max_iterations=300)
silhouette_kmeans_plus = silhouette_score(dataset_2, labels_kmeans_plus) if len(np.unique(labels_kmeans_plus)) > 1 else -1
dbscan = DBSCAN(eps=0.05, min_samples=5) 
labels_dbscan = dbscan.fit_predict(dataset_2)
silhouette_dbscan = silhouette_score(dataset_2, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else -1

print(f"Silhouette KMeans (random initialization): {silhouette_kmeans}")
print(f"Silhouette KMeans++: {silhouette_kmeans_plus}")
print(f"Silhouette DBSCAN: {silhouette_dbscan}")


# %% [markdown]
# KMeans and KMeans++ both achieved a high score of 0.7244, showing well-defined clusters with little difference between the two methods.
# DBSCAN produced a much lower score of 0.0796, indicating that it struggled to form meaningful clusters.

# %% [markdown]
# __________________________

# %% [markdown]
# ### **Elbow Method, Silhouette Score and K-distance for Dataset_3**

# %%
from scipy.spatial.distance import cdist

wcss = []
for k in range(1, 11):  # Try from 1 to 10 clusters
    centroids, labels = kmeans(dataset_3, num_clusters=k)  
    distances = cdist(dataset_3, centroids, metric='euclidean') 
    wcss.append(sum(np.min(distances, axis=1)**2)) 

# Plot the Elbow Method
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')  
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')  
plt.title('Elbow Method')  
plt.show()


# %% [markdown]
# The "elbow" point is around 4 clusters, suggesting that 4 is an optimal number of clusters for KMeans.

# %%
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_scores = []

for k in range(2, 11):  
    centroids, labels = kmeans(dataset_3, num_clusters=k) 
    score = silhouette_score(dataset_3, labels) 
    silhouette_scores.append(score)

# Plot the Silhouette Scores
plt.plot(range(2, 11), silhouette_scores, marker='o')  
plt.xlabel('Number of Clusters')  
plt.ylabel('Silhouette Score')  
plt.title('Silhouette Method')  
plt.show()


# %% [markdown]
# The silhouette score increases steadily, peaking at 10 clusters, indicating better-defined clusters as the number increases.

# %%
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# Fit the nearest neighbors model to the dataset
nearest_neighbors = NearestNeighbors(n_neighbors=5)  
nearest_neighbors.fit(dataset_3)
distances, indices = nearest_neighbors.kneighbors(dataset_3)  

# Sort distances for the k-distance graph
distances = np.sort(distances[:, -1])  
plt.plot(distances)  
plt.title("k-Distance Graph")  
plt.xlabel("Points sorted by distance")  
plt.ylabel("k-distance")  
plt.show()


# %% [markdown]
# The "knee" suggests the eps value for DBSCAN should be chosen around 1.5–2.0. This is the threshold for forming dense clusters.

# %%
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

num_clusters = 3
centroids_kmeans, labels_kmeans = kmeans(dataset_3, num_clusters, max_iterations=300)
silhouette_kmeans = silhouette_score(dataset_3, labels_kmeans) if len(np.unique(labels_kmeans)) > 1 else -1
centroids_kmeans_plus, labels_kmeans_plus = kmeans_with_kmeans_plus_plus(dataset_3, num_clusters, max_iterations=300)
silhouette_kmeans_plus = silhouette_score(dataset_3, labels_kmeans_plus) if len(np.unique(labels_kmeans_plus)) > 1 else -1

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)  # Modify the parameters based on the dataset
labels_dbscan = dbscan.fit_predict(dataset_3)

# Calculate the Silhouette Score for DBSCAN
silhouette_dbscan = silhouette_score(dataset_3, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else -1

# Print the results
print(f"Silhouette KMeans (random initialization): {silhouette_kmeans}")
print(f"Silhouette KMeans++: {silhouette_kmeans_plus}")
print(f"Silhouette DBSCAN: {silhouette_dbscan}")


# %% [markdown]
# KMeans has a score of 0.2145, showing poor clustering quality.
# 
# KMeans++ has a score of 0.1971, slightly worse than random initialization, indicating similar clustering issues.
# 
# DBSCAN: Score of 0.3436, the highest among the three, suggesting better-defined clusters, likely due to its ability to handle noise and varying densities.
# DBSCAN performs better for this dataset.

# %% [markdown]
# ________

# %% [markdown]
# ### 2.4 Best Algorithm
# 

# %% [markdown]
# ### 2.4.1

# %%
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np

# List of datasets to evaluate
datasets = [dataset_1, dataset_2, dataset_3]

# Number of clusters for each dataset
num_clusters_1 = 6  # For dataset_1
num_clusters_2_3 = 3  # For dataset_2 and dataset_3

# Evaluation metrics
metrics = {
    "silhouette": silhouette_score,  # Measures how well-separated clusters are
    "calinski_harabasz": calinski_harabasz_score,  # Higher score means better-defined clusters
    "davies_bouldin": davies_bouldin_score  # Lower score means better clustering
}

results = []

for i, data in enumerate(datasets):
    print(f"Dataset {i+1}") 
    dataset_results = {}  
    
    # Assign the number of clusters based on the dataset
    if i+1 == 1:
        num_clusters = num_clusters_1  # Use 6 clusters for dataset_1
    else:
        num_clusters = num_clusters_2_3  # Use 3 clusters for dataset_2 and dataset_3
    
    # Apply KMeans with random initialization
    _, labels_kmeans = kmeans(data, num_clusters)
    
    # Apply KMeans++ with advanced initialization
    _, labels_kmeans_plus = kmeans_with_kmeans_plus_plus(data, num_clusters)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Parameters for DBSCAN
    labels_dbscan = dbscan.fit_predict(data)  # Cluster assignments
    
    # Calculate metrics for each clustering method
    for name, metric in metrics.items():
        dataset_results[name] = {
            "KMeans": metric(data, labels_kmeans),  # Metric for KMeans
            "KMeans++": metric(data, labels_kmeans_plus),  # Metric for KMeans++
            "DBSCAN": metric(data, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else None  # Metric for DBSCAN (if multiple clusters exist)
        }
    
    
    results.append(dataset_results)

    for name, scores in dataset_results.items():
        print(f"  {name}:")  
        for algo, score in scores.items():
            print(f"    {algo}: {score:.4f}" if score is not None else f"    {algo}: N/A")  # Print the score or N/A if not applicable
    print("-" * 50)  

# %% [markdown]
# ### Dataset 1:
# 
# **Silhouette Score** :
# 
# DBSCAN (0.4828) has the highest score, indicating better cluster separation compared to KMeans (0.4594) and KMeans++ (0.4345).
# 
# **Calinski-Harabasz Score**:
# 
# KMeans (12039.4441) outperforms both KMeans++ and DBSCAN, suggesting better-defined clusters in terms of dispersion and separation.
# 
# **Davies-Bouldin Score**:
# 
# KMeans (0.9098) and DBSCAN (0.9210) perform similarly, while KMeans++ performs slightly worse (1.0860) due to less compact clusters.
# 
# ### Dataset 2:
# 
# **Silhouette Score**:
# 
# KMeans and KMeans++ (0.7244) achieve the same score, indicating identical clustering quality.
# DBSCAN did not produce multiple clusters (N/A), possibly due to unsuitable parameters.
# 
# **Calinski-Harabasz Score**:
# 
# KMeans and KMeans++ (90766.6250) show very high scores, confirming the clusters are well-defined and separated.
# 
# **Davies-Bouldin Score**:
# 
# KMeans (0.4106) is better, indicating more compact and separated clusters compared to DBSCAN (N/A).
# 
# ### Dataset 3:
# 
# **Silhouette Score**:
# 
# KMeans (2.0050) has the highest score, showing superior clustering quality, followed by KMeans++ (1.9503) and DBSCAN (1.3272).
# 
# **Calinski-Harabasz Score**:
# 
# Similar trend as the silhouette score, with KMeans outperforming the other methods.
# 
# **Davies-Bouldin Score**:
# 
# KMeans performs best with the lowest value, while DBSCAN is the least effective for compact and separated clusters.

# %% [markdown]
# ### 2.4.2

# %% [markdown]
# ### Metrics to Assess Clustering Quality
# 
# #### 1. Silhouette Score
# 
# The Silhouette Score measures how well each data point fits within its assigned cluster compared to other clusters.
# - **Formula**:
#   $$
#   S = \frac{b - a}{\max(a, b)}
#   $$
#   Where:
#   - $a$: Average distance of a point to other points in its own cluster.
#   - $b$: Average distance of a point to points in the nearest neighboring cluster (inter-cluster distance).
# - **Range**: 
#   - $1$: Perfect clustering (points are well-separated).
#   - $0$: Overlapping clusters.
#   - Negative values: Poor clustering (points are misclassified).
#  Measures both **compactness** (how close points are within the same cluster) and **separation** (how far clusters are from each other).
# 
# ---
# 
# #### 2. Calinski-Harabasz Score
# 
# This score measures the ratio of between-cluster variance to within-cluster variance, indicating how well the clusters are defined.
# - **Formula**:
#   $$
#   CH = \frac{\text{trace between-cluster scatter matrix}}{\text{trace within-cluster scatter matrix}} \times \frac{n-k}{k-1}
#   $$
#   Where:
#   - $n$: Total number of data points.
#   - $k$: Number of clusters.
# - **Range**: Higher values indicate better-defined clusters.
# - **Evaluation**: Focuses on cluster **separation** and **compactness** by comparing cluster variance.
# 
# ---
# 
# #### 3. Davies-Bouldin Score
# This score evaluates the average similarity between each cluster and the most similar other cluster. Lower values indicate better clustering.
# - **Formula**:
#   $$
#   DB = \frac{1}{k} \sum_{i=1}^{k} \max_{i \neq j} \frac{S_i + S_j}{M_{ij}}
#   $$
#   Where:
#   - $S_i$: Average intra-cluster distance (compactness) for cluster $i$.
#   - $M_{ij}$: Distance between the centroids of clusters $i$ and $j$ (separation).
# - **Range**: Lower values are better, as they indicate compact clusters that are far apart.
# - **Evaluation**: Focuses on both **compactness** and **separation**, penalizing overlapping clusters.
# 
# ---
# 
# - **Silhouette Score**: Evaluates how well points are clustered, balancing compactness and separation.
# - **Calinski-Harabasz Score**: Assesses the ratio of cluster separation to dispersion.
# - **Davies-Bouldin Score**: Penalizes clusters that are not compact or well-separated.
# 
# 
# 

# %% [markdown]
# ### 2.4.3

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Copy the standardized dataset
data_reduced = dataset_1.copy()

# Custom KMeans clustering
num_clusters = 6  # Number of clusters
centroids, kmeans_labels = kmeans(data_reduced, num_clusters)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_reduced)

# 3D Visualization of clustering results
fig = plt.figure(figsize=(14, 6))

# KMeans clustering plot
ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(
    data_reduced.iloc[:, -1], data_reduced.iloc[:, -2], data_reduced.iloc[:, -3],
    c=kmeans_labels, cmap='viridis', s=20
)
ax.set_title("KMeans Clustering")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# DBSCAN clustering plot
ax = fig.add_subplot(122, projection='3d')
scatter = ax.scatter(
    data_reduced.iloc[:, -1], data_reduced.iloc[:, -2], data_reduced.iloc[:, -3],
    c=dbscan_labels, cmap='viridis', s=20
)
ax.set_title("DBSCAN Clustering")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add shared colorbar and show the plot
plt.colorbar(scatter, ax=fig.get_axes(), shrink=0.5)
plt.tight_layout()
plt.show()


# %% [markdown]
# 
# **KMeans Clustering** (Left Plot):
# Forms 6 clusters, with clear separation in some regions but visible overlap in denser areas.
# Performs well for structured, spherical clusters but struggles with overlapping regions.
# 
# **DBSCAN Clustering** (Right Plot):
# Groups points based on density, with some points classified as noise.
# Handles arbitrary-shaped clusters but identifies fewer distinct groups, due to parameter settings (eps, min_samples).
# Evaluation Metrics:
# 
# 
# Overall KMeans is better for spherical clusters but struggles with overlapping.
# DBSCAN is better for handling irregular shapes and noise but depends heavily on parameter tuning.
# Metric scores will clarify which algorithm is more effective for this dataset.

# %% [markdown]
# ## **3. Bonus Question**

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# I used Principal Component Analysis (PCA) to reduce the high-dimensional movie features (ratings, tag relevance scores, and number of ratings) into two principal components (PC1 and PC2). These components were selected because they capture the largest variance in the data, allowing for better visualization and clustering in 2D. PCA helps to retain the most significant information while simplifying the dataset, making it easier to identify patterns and groupings.

# %%
# step1: Combine relevant data
genome_data = genome_scores.merge(genome_tags, on="tagId")  # Add human-readable tags
tag_features = genome_data.pivot_table(index="movieId", columns="tag", values="relevance", fill_value=0)
rating_features = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "size")
).reset_index()

# Merge all features
movie_features = movie.merge(tag_features, on="movieId", how="left").merge(rating_features, on="movieId", how="left")
movie_features.fillna(0, inplace=True)

# Step 2: Normalize Data
features = movie_features.drop(columns=["movieId", "title", "genres"])
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Step 3: Dimensionality Reduction using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)

# Step 4: K-Means Clustering and Iteration Tracking
centroids = []
all_labels = []
n_clusters = 3

for iteration in range(10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=1, max_iter=iteration + 1)
    kmeans.fit(reduced_features)
    centroids.append(kmeans.cluster_centers_)
    all_labels.append(kmeans.labels_)

# Step 5: Visualization of Clustering Progression
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    if i < len(centroids):
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels[i], cmap='viridis', s=50, alpha=0.7)
        ax.scatter(centroids[i][:, 0], centroids[i][:, 1], c='red', marker='x', s=200)
        ax.set_title(f"Iteration {i+1}")
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by PCA Components: {explained_variance}")

# %% [markdown]
# ## **4. Algorithmic Question**

# %% [markdown]
# ### AQ a) Pseudocode exercise
# 
#     FUNCTION optimal_strategy(nums, sum_Arya = 0, sum_Mario = 0, turn = True):
#     IF nums is empty:
#         RETURN (sum_Arya > sum_Mario)
# 
#     IF it is Arya's turn (turn is True):
#         # Arya can choose the first or last number
#         take_first = CALL optimal_strategy(nums without first element, sum_Arya + first number of nums, sum_Mario, turn = False)
#         take_last = CALL optimal_strategy(nums without last element, sum_Arya + last number of nums, sum_Mario, turn = False)
# 
#         RETURN (take_first OR take_last)
#     ELSE:
#         # Mario can choose the first or last number
#         take_first = CALL optimal_strategy(nums without first element, sum_Arya, sum_Mario + first number of nums, turn = True)
#         take_last = CALL optimal_strategy(nums without last element, sum_Arya, sum_Mario + last number of nums, turn = True)
# 
#         RETURN (take_first AND take_last)
# 
#     # Main logic to test multiple cases
#     INITIALIZE testcases as a list of test cases
#       FOR each testcase in testcases:
#         IF CALL optimal_strategy(testcase):
#           PRINT "True"
#         ELSE:
#           PRINT "False"
# 

# %%
import time

# %% [markdown]
# ### AQ b)
# Python code

# %%
def optimal_strategy(nums, sum_Arya=0, sum_Mario=0, turn=True):
    # Base case: when no numbers are left
    if len(nums) == 0:

        return sum_Arya > sum_Mario

    # Recursive case
    if turn:  # Arya's turn

        # Arya chooses either the first or the last number
        take_first = optimal_strategy(nums[1:], sum_Arya + nums[0], sum_Mario, False)
        take_last = optimal_strategy(nums[:-1], sum_Arya + nums[-1], sum_Mario, False)

        return take_first or take_last
    else:  # Mario's turn
        # Mario chooses either the first or the last number
        take_first = optimal_strategy(nums[1:], sum_Arya, sum_Mario + nums[0], True)
        take_last = optimal_strategy(nums[:-1], sum_Arya, sum_Mario + nums[-1], True)

        return take_first and take_last

# Test with input lists
testcases=[]
# 1st testcase
nums = [1, 5, 2]
testcases.append(nums)
# 2nd testcase
nums = [1, 5, 233, 7]
testcases.append(nums)
start_time= time.time()
for testcase in testcases:
    if optimal_strategy(testcase):
        print(True)
    else:
        print(False)
end_time=time.time()
print(f"Execution time: {end_time - start_time} seconds")


# %% [markdown]
# ### AQ c)
# 1. Problem Definition
# The function optimal_strategy aims to determine whether Arya can win by selecting from the available numbers (nums). The numbers are arranged in an array, and Arya or Mario can choose to take either the first or the last element during their turn. The game continues until no numbers are left.
# 
# 2. Recursive Structure
# Each recursive call of the function splits into two successive calls:
# 
# One for the case where the player chooses the first number.
# Another for the case where the player chooses the last number.
# Each choice reduces the length of the array (nums) by 1, but generates two new branches of computation. This behavior forms a binary decision tree.
# 
# 3. Problem Size
# If the input array has a length of n, then:
# 
# At the initial level, there are 2 choices (first or last number).
# At each step, the number of possible states doubles, as each generates two further recursive calls.
# The total number of calls grows as
# 2^n, which represents exponential time complexity.
# 

# %% [markdown]
# ### 3d) Pseudocode of optimal solution
# 

# %% [markdown]
# **Function**: OptimalStrategy(nums)<br>
# **Input**: nums (array of integers)<br>
# **Output**: Boolean (True if Arya can ensure a win, False otherwise)<br>
# **Steps**:
# Let n be the length of nums.
# 
# **Initialize** DP Table:
# 
# **Create** a 2D list `dp` of size n x n initialized to False.
# `dp[i][j]` will store Arya has a winning strategy for the subarray `nums[i:j+1]`.<br>
# Compute **Prefix** Sums:
# 
# **Create an array** `sum_nums` of size `n + 1` initialized to 0.<br>
# **For each** i from 0 to n - 1:
# `sum_nums[i + 1] = sum_nums[i] + nums[i].`<br>
# **Base Case**: Subarrays of Length 1
# 
# **For each i from 0 to n - 1**:<br>
# `Set dp[i][i] = True` (Arya always wins if there's only one element).
# Iterate Over subarrays:
# 
# **For each** length from 2 to n:<br>
# &emsp;**For each** starting index i from 0 to n - length:<br>
# **Let** j = `i + length - 1` (end index of the subarray).<br>
# **Calculate** `take_first = NOT dp[i + 1][j]` (Arya takes nums[i]).<br>
# **Calculate** `take_last = NOT dp[i][j - 1]` (Arya takes nums[j]).<br>
# **Set** `dp[i][j] = take_first OR take_last`.<br>
# **Return** Result:
# 
# **Return** the value of `dp[0][n - 1]` (Arya's winning strategy for the entire array).

# %% [markdown]
# ### Time complexity of optimal algorithm:
# 
# 
# 1.   Creation matrix `n*n` is `O(n^2)` time complexity
# 2.   `For each i from o to n-1` is `O(n)` time complexity
# 3.   `For each i from o to n-1`, set `dp[i][i]=True` is `O(n)` time complexity
# 4.   Iteration over subarrays takes up to `O(n*n)= O(n^2)` time complexity, because of 2 loops(outer and inner)
# **Total time complexity** is: `O(n^2)`+ `O(n)`+ `O(n)`+`O(n^2)`=`O(n^2)`
# 
# 
# 

# %% [markdown]
# ### 3e) Python code of **optimal** solution

# %%
def optimal_strategy(nums):
    n = len(nums)
    # Create a DP table
    dp = [[False] * n for _ in range(n)]

    # Sum array to calculate the cumulative sum of nums
    sum_nums = [0] * (n + 1)
    for i in range(n):
        sum_nums[i + 1] = sum_nums[i] + nums[i]

    # Fill DP for subarrays of length 1 (base case)
    for i in range(n):
        dp[i][i] = True  # Arya always wins if there's one element, as it's her turn

    # check of subarrays
    for length in range(2, n + 1):  # length = 2 to n
        for i in range(n - length + 1):
            j = i + length - 1  # End of the current subarray
            #choose nums for Arya
            take_first = not dp[i + 1][j]  # Arya takes nums[i], Mario's turn
            take_last = not dp[i][j - 1]  # Arya takes nums[j], Mario's turn
            dp[i][j] = take_first or take_last

    # Final answer for the entire array
    return dp[0][n - 1]

# Test with input lists
testcases = [
    [1, 5, 2],         # Test case 1
    [1, 5, 233, 7],    # Test case 2
]
start_time= time.time()
for testcase in testcases:
    if optimal_strategy(testcase):
        print(True)
    else:
        print(False)
end_time=time.time()
print(f"Execution time: {end_time - start_time} seconds")


# %% [markdown]
# *Execution time* of **exponentional** algorithm equals to 0.002 seconds<br>
# *Execution time* of **polynomial** algorithm equals to: 0.0003 seconds<br>
# It shows us how is faster polynomial algorithm than exponentional one

# %% [markdown]
# ### 3f) Python code written by LLM

# %%
def optimal_strategy(nums):
    n = len(nums)

    # Previous and current rows of the DP table
    prev = [False] * n
    curr = [False] * n

    # Base case: Subarrays of length 1
    for i in range(n):
        prev[i] = True  # Arya always wins if there's one element

    # Fill DP for subarrays of increasing lengths
    for length in range(2, n + 1):  # length = 2 to n
        for i in range(n - length + 1):
            j = i + length - 1  # End of the current subarray

            # Arya's choice
            take_first = not prev[i + 1]  # Arya takes nums[i], Mario's turn
            take_last = not prev[i]  # Arya takes nums[j], Mario's turn

            # Update current row
            curr[i] = take_first or take_last

        # Move to the next row
        prev, curr = curr, prev

    # Final answer for the entire array
    return prev[0]

# Test with input lists
testcases = [
    [1, 5, 2],         # Test case 1
    [1, 5, 233, 7],    # Test case 2
]

for testcase in testcases:
    print(optimal_strategy(testcase))


# %% [markdown]
# Actually it didn't reduce time complexity, only space complexity instead of using table with dimensions `n*n` it used only 2 arrays with length equals to n, algorithm had done a good job because it is similar to optimal solution i added here, also it improved my solution because of using less auxiliary space


