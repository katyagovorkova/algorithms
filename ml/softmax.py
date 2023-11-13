import numpy as np

def softmax(x):
    """
        Compute softmax for each x values
        >>> input_vector = [3, 4.5, -1]
        >>> softmax(input_vector)
        >>> array([0.18181803, 0.81485186, 0.00333011])
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)