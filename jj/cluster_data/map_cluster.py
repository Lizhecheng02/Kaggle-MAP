import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from collections import defaultdict
import random


def calculate_similarity_matrix(texts, similarity_func=fuzz.ratio):
    """
    Calculate pairwise similarity matrix for a list of texts.
    Returns distance matrix (1 - similarity/100) for clustering.
    """
    n = len(texts)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_func(texts[i], texts[j])
            distance = 1 - (similarity / 100.0)  # Convert to distance
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def cluster_similar_texts(texts, similarity_threshold=0.8, min_samples=1):
    """
    Cluster similar texts using DBSCAN.

    Args:
        texts: List of text strings
        similarity_threshold: Minimum similarity (0-1) to be in same cluster
        min_samples: Minimum samples in a cluster for DBSCAN

    Returns:
        cluster_labels: Array of cluster labels (-1 for noise)
    """
    if len(texts) <= 1:
        return np.array([0] * len(texts))

    # Calculate distance matrix
    distance_matrix = calculate_similarity_matrix(texts)

    # Convert similarity threshold to distance threshold
    eps = 1 - similarity_threshold

    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = clustering.fit_predict(distance_matrix)

    return cluster_labels


def deduplicate_by_question(
    df,
    similarity_threshold=0.8,
    sampling_strategy="random",
    return_all_with_clusters=False,
):
    """
    Deduplicate student explanations within each question group.

    Args:
        df: DataFrame with 'QuestionId' and 'StudentExplanation' columns
        similarity_threshold: Similarity threshold for clustering (0-1)
        sampling_strategy: 'random', 'longest', or 'shortest'
        return_all_with_clusters: If True, return all rows with cluster column instead of sampling

    Returns:
        If return_all_with_clusters=False: Deduplicated DataFrame (one sample per cluster)
        If return_all_with_clusters=True: Original DataFrame with additional 'ClusterId' column
    """
    if return_all_with_clusters:
        return add_cluster_labels(df, similarity_threshold)

    deduplicated_rows = []

    for question_id, group in df.groupby("QuestionId"):
        texts = group["StudentExplanation"].tolist()
        indices = group.index.tolist()

        if len(texts) <= 1:
            deduplicated_rows.extend(indices)
            continue

        # Cluster similar texts
        cluster_labels = cluster_similar_texts(texts, similarity_threshold)

        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)

        # Sample one from each cluster
        for cluster_indices in clusters.values():
            if sampling_strategy == "random":
                chosen_idx = random.choice(cluster_indices)
            elif sampling_strategy == "longest":
                lengths = [len(texts[i]) for i in cluster_indices]
                chosen_idx = cluster_indices[np.argmax(lengths)]
            elif sampling_strategy == "shortest":
                lengths = [len(texts[i]) for i in cluster_indices]
                chosen_idx = cluster_indices[np.argmin(lengths)]

            original_idx = indices[chosen_idx]
            deduplicated_rows.append(original_idx)

    return df.loc[deduplicated_rows].reset_index(drop=True)


def add_cluster_labels(df, similarity_threshold=0.8):
    """
    Add cluster labels to all rows in the dataframe.

    Args:
        df: DataFrame with 'QuestionId' and 'StudentExplanation' columns
        similarity_threshold: Similarity threshold for clustering (0-1)

    Returns:
        DataFrame with additional 'ClusterId' column
    """
    result_df = df.copy()
    cluster_ids = []

    for question_id, group in df.groupby("QuestionId"):
        texts = group["StudentExplanation"].tolist()

        if len(texts) <= 1:
            # Single item gets its own cluster
            cluster_ids.extend([f"{question_id}_cluster_0"])
            continue

        # Cluster similar texts
        cluster_labels = cluster_similar_texts(texts, similarity_threshold)

        # Create meaningful cluster IDs
        question_cluster_ids = [
            f"{question_id}_cluster_{label}" for label in cluster_labels
        ]
        cluster_ids.extend(question_cluster_ids)

    result_df["ClusterId"] = cluster_ids
    return result_df


def analyze_duplicates(df, similarity_threshold=0.8):
    """
    Analyze duplicate patterns in the dataset.
    """
    stats = {
        "original_count": len(df),
        "questions": df["QuestionId"].nunique(),
        "clusters_per_question": [],
        "items_per_cluster": [],
    }

    for question_id, group in df.groupby("QuestionId"):
        texts = group["StudentExplanation"].tolist()

        if len(texts) <= 1:
            stats["clusters_per_question"].append(1)
            stats["items_per_cluster"].append(1)
            continue

        cluster_labels = cluster_similar_texts(texts, similarity_threshold)
        unique_clusters = len(set(cluster_labels))
        stats["clusters_per_question"].append(unique_clusters)

        # Count items per cluster
        clusters = defaultdict(int)
        for label in cluster_labels:
            clusters[label] += 1

        stats["items_per_cluster"].extend(clusters.values())

    return stats


# Example usage and advanced techniques
def advanced_similarity_matching(
    texts, methods=["ratio", "partial_ratio", "token_sort_ratio"]
):
    """
    Use multiple RapidFuzz methods and combine scores.
    """
    n = len(texts)
    combined_distances = np.zeros((n, n))

    method_funcs = {
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
    }

    for i in range(n):
        for j in range(i + 1, n):
            scores = []
            for method in methods:
                if method in method_funcs:
                    score = method_funcs[method](texts[i], texts[j])
                    scores.append(score)

            # Average the scores
            avg_similarity = np.mean(scores)
            distance = 1 - (avg_similarity / 100.0)
            combined_distances[i, j] = distance
            combined_distances[j, i] = distance

    return combined_distances


def hierarchical_clustering_approach(texts, similarity_threshold=0.8):
    """
    Alternative approach using hierarchical clustering.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    if len(texts) <= 1:
        return np.array([0] * len(texts))

    # Calculate distance matrix
    distance_matrix = calculate_similarity_matrix(texts)

    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method="average")

    # Get clusters at specified threshold
    distance_threshold = 1 - similarity_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")

    # Convert to 0-based indexing
    return cluster_labels - 1


def sample_from_clusters(
    df_with_clusters, n_samples_per_cluster=1, sampling_strategy="random"
):
    """
    Sample from each cluster in a dataframe that has cluster labels.

    Args:
        df_with_clusters: DataFrame with 'ClusterId' column
        n_samples_per_cluster: Number of samples to take from each cluster
        sampling_strategy: 'random', 'longest', 'shortest', or 'all'

    Returns:
        Sampled DataFrame
    """
    if sampling_strategy == "all" or n_samples_per_cluster == -1:
        return df_with_clusters

    sampled_rows = []

    for cluster_id, cluster_group in df_with_clusters.groupby("ClusterId"):
        cluster_size = len(cluster_group)
        n_to_sample = min(n_samples_per_cluster, cluster_size)

        if sampling_strategy == "random":
            sampled = cluster_group.sample(n=n_to_sample, random_state=42)
        elif sampling_strategy == "longest":
            # Calculate lengths and sort by them
            cluster_copy = cluster_group.copy()
            cluster_copy["_text_length"] = cluster_copy["StudentExplanation"].str.len()
            sampled = cluster_copy.nlargest(n_to_sample, "_text_length").drop(
                "_text_length", axis=1
            )
        elif sampling_strategy == "shortest":
            # Calculate lengths and sort by them
            cluster_copy = cluster_group.copy()
            cluster_copy["_text_length"] = cluster_copy["StudentExplanation"].str.len()
            sampled = cluster_copy.nsmallest(n_to_sample, "_text_length").drop(
                "_text_length", axis=1
            )

        sampled_rows.append(sampled)

    return pd.concat(sampled_rows).reset_index(drop=True)


def get_cluster_stats(df_with_clusters):
    """
    Get detailed statistics about clusters.
    """
    cluster_stats = (
        df_with_clusters.groupby(["QuestionId", "ClusterId"])
        .agg(
            {
                "StudentExplanation": [
                    "count",
                    lambda x: x.str.len().mean(),
                    lambda x: x.str.len().std(),
                ]
            }
        )
        .round(2)
    )

    cluster_stats.columns = ["ClusterSize", "AvgLength", "StdLength"]
    cluster_stats = cluster_stats.reset_index()

    return cluster_stats
