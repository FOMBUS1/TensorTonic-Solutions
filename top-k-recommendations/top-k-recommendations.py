import numpy as np

def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    scores = np.array(scores)
    inx_score = {inx: score for inx, score in enumerate(scores) if inx not in rated_indices}

    return sorted(inx_score, key=lambda x: inx_score[x], reverse=True)[:k]