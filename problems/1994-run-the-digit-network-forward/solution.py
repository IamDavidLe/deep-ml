import numpy as np
import math


def digit_network_forward(image, weights, biases):
    """Run an image through a fully connected sigmoid network.

    Returns (output activations rounded to 4 decimals, predicted digit).
    """
    
    a = np.array(image, dtype = float).flatten()
    for W, b in zip(weights, biases):
        W = np.array(W, dtype = float)
        b = np.array(b, dtype = float)
        z = W @ a + b 
        a = 1 / (1 + np.exp(-z)) # the output of the previous layer is the input of the latter one
    
    predicted_digit = int(np.argmax(a)) # return the index that has the largest value
    return (np.round(a, 4).tolist(), predicted_digit)

