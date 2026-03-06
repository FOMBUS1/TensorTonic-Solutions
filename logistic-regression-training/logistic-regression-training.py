import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    
    w = np.zeros(X.shape[1])
    b = 0.0
    
    for step in range(steps): 
        p = _sigmoid(X@w + b)
    
        W_grad = (X.T@(p - y)) / len(y)
        b_grad = (p-y).mean()
        w -= lr*W_grad
        b -= lr*b_grad
    return w,b