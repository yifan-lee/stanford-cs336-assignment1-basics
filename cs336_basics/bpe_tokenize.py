import regex as re

from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN = tuple[bytes]
PAIR = tuple[bytes, bytes]


def update_token_freqs(
    text_segment: str,
    token_freqs: dict[TOKEN, int],
):
    matches = re.finditer(PAT, text_segment)
    for m in matches:
        token = m.group()
        token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))
        token_freqs[token_bytes] += 1 

def get_pairs(
    token: TOKEN
) -> list[PAIR]:
    if len(token) < 2:
        return []
    return [(token[i], token[i+1]) for i in range(len(token)-1)]

def update_pair_freqs(
    token_freqs: dict[TOKEN, int],
    pair_freqs: dict[PAIR, int],
):
    for token, freq in token_freqs.items():
        if len(token) < 2:
            continue
        pairs = get_pairs(token)
        for p in pairs:
            pair_freqs[p] += freq

def update_pair2token(
    token_freqs: dict[TOKEN, int],
    pair_to_token: dict[PAIR, set[TOKEN]],
):
    for token, freq in token_freqs.items():
        if len(token) < 2:
            continue
        pairs = get_pairs(token)
        for p in pairs:
            pair_to_token[p].add(token)

def update_token2pair(
    token_freqs: dict[TOKEN, int],
    token_to_pair: dict[TOKEN, list[PAIR]],
):
    for token, freq in token_freqs.items():
        if len(token) < 2:
            continue
        pairs = get_pairs(token)
        token_to_pair[token] = pairs

def get_most_frequent_pair(
    pair_freqs: dict[PAIR, int],
) -> PAIR:
    return max(pair_freqs.keys(), key=lambda k: (pair_freqs[k], k))

def update_vocab(
    new_id: int,
    best_pair: PAIR,
    vocab: dict[int, bytes],
):
    new_vocab = best_pair[0] + best_pair[1]
    vocab[new_id] = new_vocab

def update_vocab_inverse(
    new_id: int,
    best_pair: PAIR,
    vocab_inverse: dict[bytes, int],
):
    new_vocab = best_pair[0] + best_pair[1]
    vocab_inverse[new_vocab] = new_id

def update_merges(
    best_pair: PAIR,
    merges: list[PAIR],
):
    merges.append(best_pair)

def update_all(
    best_pair: PAIR,
    pair_to_token: dict[PAIR, set[TOKEN]],
    token_to_pair: dict[TOKEN, list[PAIR]],
    token_freqs: dict[TOKEN, int],
    pair_freqs: dict[PAIR, int],
):
    affected_tokens = list(pair_to_token[best_pair])
    merged_bytes = best_pair[0] + best_pair[1]

    for token in affected_tokens:
        # get new token
        i=0
        new_token = []
        while(i<(len(token))):
            if (i < len(token) - 1) and (token[i] == best_pair[0]) and (token[i+1] == best_pair[1]):
                new_token.append(merged_bytes)
                i = i + 2
            else:
                new_token.append(token[i])
                i = i + 1
        new_token = tuple(new_token)


        ## update pair_to_token
        new_pairs = get_pairs(new_token)
        affected_pairs = token_to_pair[token]
        for pair in affected_pairs:
            pair_to_token[pair].discard(token)
        for pair in new_pairs:
            pair_to_token[pair].add(new_token)


        ## update token_to_pair
        token_to_pair.pop(token)
        token_to_pair[new_token] = new_pairs


        ## update pair_freqs
        for pair in affected_pairs:
            pair_freqs[pair] -= token_freqs[token]
        for pair in new_pairs:
            pair_freqs[pair] += token_freqs[token]


        ## update token_freqs
        origin_freq = token_freqs.pop(token)
        token_freqs[new_token] = origin_freq
        

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    vocab = {i:bytes([i]) for i in range(256)}
    vocab_inverse = {v:k for k,v in vocab.items()}
    # Add special tokens
    for st in special_tokens:
        new_id = len(vocab)
        st_bytes = st.encode("utf-8")
        vocab[new_id] = st_bytes
        vocab_inverse[st_bytes] = new_id
    
    merges = []

    with open(input_path, 'r', encoding='utf-8') as file:
        chunks = file.read()

    special_pat = "|".join(re.escape(st) for st in special_tokens)
    segments = re.split(special_pat, chunks)


    token_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    pair_to_token = defaultdict(set)
    token_to_pair = defaultdict(list)

    for text_segment in segments:

        ## Initiate token_freqs
        update_token_freqs(text_segment, token_freqs)

    # Initiate pair_freqs, pair_to_token, and token_to_pair
    update_pair_freqs(token_freqs, pair_freqs)
    update_pair2token(token_freqs, pair_to_token)
    update_token2pair(token_freqs, token_to_pair)


    num_merges = vocab_size - 256 - len(special_tokens)
    for i in range(num_merges):
        ### find most frequent pair
        if not pair_freqs:
            break
        best_pair = get_most_frequent_pair(pair_freqs)

        new_id = len(vocab)

        ## update vocab
        update_vocab(new_id, best_pair, vocab)

        ## update vocab_inverse
        update_vocab_inverse(new_id, best_pair, vocab_inverse)

        ## update merges
        update_merges(best_pair, merges)

        update_all(best_pair, pair_to_token, token_to_pair, token_freqs, pair_freqs)

    

    return vocab, merges
    

if __name__ == "__main__":

    input_path: str = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size: int = 260
    special_tokens: list[str] = ['<|endoftext|>']


    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        
    print(merges)