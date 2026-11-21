def wordpiece_tokenize(text: str, vocab: dict,
                       unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    tokens = []
    w = text.lower().split()
    for word in w:
        n = len(word)
        if n > max_word_length:
            tokens.append(unk_token)
            continue

        pieces = []
        start = 0
        while start < n:
            end = n
            cur_substr = None
            while start < end:
                substr = word[start : end]
                if start > 0:
                    substr = "##" + substr
    
                if substr in vocab:
                    cur_substr = substr
                    break
                        
                end -= 1
            if cur_substr is None:
                pieces = [unk_token]
                break
            pieces.append(cur_substr)
            start = end
        tokens.extend(pieces)
    return tokens