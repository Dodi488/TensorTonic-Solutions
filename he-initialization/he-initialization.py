import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    L = math.sqrt(6 / fan_in)
    return [[(2 * val - 1) * L for val in row] for row in W]