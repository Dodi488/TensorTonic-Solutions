def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    # Write code here
    text_length = len(tokens)
    
    if overlap == 0:
        num_chunks = -(-text_length // chunk_size)
    else:
        num_chunks = -(-(text_length - overlap) // (chunk_size - overlap))
    
    stride = chunk_size - overlap
    final = []
    for i in range(num_chunks):
        final.append(tokens[i * stride : i * stride + chunk_size])
    return final