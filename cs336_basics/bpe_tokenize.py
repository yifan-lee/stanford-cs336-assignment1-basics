import pickle
import regex as re
import time
from tqdm import tqdm

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
    print("Start Training BPE...")
    start_total = time.time()

    print("Initialize Vocab and Merges...")
    vocab = {i:bytes([i]) for i in range(256)}
    vocab_inverse = {v:k for k,v in vocab.items()}
    # Add special tokens
    for st in special_tokens:
        new_id = len(vocab)
        st_bytes = st.encode("utf-8")
        vocab[new_id] = st_bytes
        vocab_inverse[st_bytes] = new_id
    
    merges = []

    print("Read Input File...")
    t0 = time.time()
    with open(input_path, 'r', encoding='utf-8') as file:
        chunks = file.read()
    print(f"Read Input File took {time.time() - t0:.2f}s")

    t0 = time.time()
    special_pat = "|".join(re.escape(st) for st in special_tokens)
    segments = re.split(special_pat, chunks)
    print(f"Regex Split took {time.time() - t0:.2f}s")


    token_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    pair_to_token = defaultdict(set)
    token_to_pair = defaultdict(list)

    ## Initiate token_freqs
    print("Update Token Freq...")
    t0 = time.time()
    for text_segment in tqdm(segments, desc="Token Freqs"):
        update_token_freqs(text_segment, token_freqs)
    print(f"Update Token Freq took {time.time() - t0:.2f}s")

    ## Initiate pair_freqs, pair_to_token, and token_to_pair
    print("Update Pair Freq...")
    t0 = time.time()
    update_pair_freqs(token_freqs, pair_freqs)
    print(f"Update Pair Freq took {time.time() - t0:.2f}s")

    print("Update Pair to Token...")
    t0 = time.time()
    update_pair2token(token_freqs, pair_to_token)
    print(f"Update Pair to Token took {time.time() - t0:.2f}s")

    print("Update Token to Pair...")
    t0 = time.time()
    update_token2pair(token_freqs, token_to_pair)
    print(f"Update Token to Pair took {time.time() - t0:.2f}s")

    print("Start Merging...")
    t0 = time.time()
    num_merges = vocab_size - 256 - len(special_tokens)
    for i in tqdm(range(num_merges), desc="Merging"):
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
    print(f"Merging took {time.time() - t0:.2f}s")
    print(f"Total Training took {time.time() - start_total:.2f}s")

    return vocab, merges
    

if __name__ == "__main__":

    data_group = "train"
    file_name = f"TinyStoriesV2-GPT4-{data_group}.txt"
    input_path = f"data/{file_name}"
    vocab_size = 260
    special_tokens = ['<|endoftext|>']

    

    vocab_path = f"outputs/{file_name.split(".")[0]}-vocab.pkl"
    merges_path = f"outputs/{file_name.split(".")[0]}-merge.pkl"


    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)