def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    # Write code here
    values = {}
    counts = {}

    for item, target in zip(categories, targets):
        counts[item] = counts.get(item, 0) + 1
        values[item] = values.get(item, 0) + target

    means = {i: values[i] / counts[i] for i in counts}
    return [means[i] for i in categories]