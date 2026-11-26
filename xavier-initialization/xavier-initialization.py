import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    a = math.sqrt(6 / (fan_in + fan_out))
    return [[(2 * val - 1) * a for val in row] for row in W]