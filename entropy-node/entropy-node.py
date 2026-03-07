import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    p = np.unique(y, return_counts=True)
    p = [count / len(y) for count in p[1]]
    
    return -(p*np.log2(p)).sum()