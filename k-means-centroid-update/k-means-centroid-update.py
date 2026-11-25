def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    final = []
    for i in range(k):
        data = [p for p, t in zip(points, assignments) if t == i]
        if not data:
            final.append([0.0 for i in range(len(points[i]))])
        else:
            final.append([sum(j) / len(data) for j in zip(*data)])
    return final