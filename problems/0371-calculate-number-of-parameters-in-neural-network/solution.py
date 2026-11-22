def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    

    # Your code here
    params = 0

    for layer in layers:
        if layer['type'] == 'dense':
            weights = layer['input_size'] * layer['output_size']
            biases = layer['output_size'] if layer.get('bias', True) else 0

            params += weights + biases

        elif layer['type'] == 'conv2d':
            weights = (
                layer['in_channels']
                * layer['out_channels']
                * layer['kernel_size'] ** 2
            )
            biases = layer['out_channels'] if layer.get('bias', True) else 0

            params += weights + biases

    return params
