def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    # Write code here
    output = []
    rotated = [list(row) for row in zip(*(r for r in predictions))]
    for i in rotated:
        output.append(max(set(i), key=i.count))
    return output