import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    # Write code here
    H = 0
    N = len(actual_tokens)
    for i in range(N):
        H += math.log(prob_distributions[i][actual_tokens[i]])

    H = -H / N
    return math.exp(H)