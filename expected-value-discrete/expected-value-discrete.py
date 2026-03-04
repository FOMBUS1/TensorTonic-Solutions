import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    if not(np.isclose(p.sum(), 1, atol=1e-6)):
        raise ValueError()

    return x@p
