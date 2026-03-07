import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.array(x)

    return np.array([np.quantile(x, qi/100) for qi in q])