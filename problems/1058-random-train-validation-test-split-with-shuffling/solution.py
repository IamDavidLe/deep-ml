import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]

    indices = np.random.default_rng(seed).permutation(n)
    shuffled = data[indices]

    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    train = shuffled[:train_end]
    validation = shuffled[train_end:validation_end]
    test = shuffled[validation_end:]

    return [train, validation, test]