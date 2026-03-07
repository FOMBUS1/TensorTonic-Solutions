import numpy as np

def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here

    items = np.array(items)
    WR = (items[:, 1] / (items[:, 1] + min_votes)) * items[:, 0] + min_votes/(items[:, 1] + min_votes) * global_mean

    return WR.tolist()
    