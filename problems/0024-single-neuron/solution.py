import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here
    probabilities = []
    for x in features:
        z = sum(xi * wi for xi, wi in zip(x, weights)) + bias
        a = 1 / (1 + math.exp(-z))
        probabilities.append(a)
    
    mse = sum((prob - lab) ** 2 for prob, lab in zip(probabilities, labels)) / len(labels)
    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)
    return probabilities, mse