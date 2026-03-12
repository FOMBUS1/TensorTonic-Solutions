def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    a = set(set_a)
    b = set(set_b)

    return 0.0 if (len(a&b) == 0 and len(a | b) == 0) else (len(a & b) / len(a | b))