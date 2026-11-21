def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    """
    Returns sentence_a, sentence_b, and is_next dictionaries in a list.
    """
    output = []
    for i in pair_specs:
        idx1 = i["doc_a"]
        idx2 = i["sent_a"]
        idx3 = i["doc_b"]
        idx4 = i["sent_b"]
        if idx2 == (idx4 - 1) and idx1 == idx3:
            is_next = 1
        else:
            is_next = 0
        output.append({"sentence_a":documents[idx1][idx2],"sentence_b":documents[idx3][idx4],"is_next":is_next})
    return output