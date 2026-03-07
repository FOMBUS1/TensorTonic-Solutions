def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    precision_k = len(set(recommended[:k]) & set(relevant)) / k
    recall_k = len(set(recommended[:k]) & set(relevant)) / len(relevant)

    return [precision_k, recall_k]