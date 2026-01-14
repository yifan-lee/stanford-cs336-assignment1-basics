import os
import pickle
import time

import regex as re

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from collections import defaultdict




PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN = tuple[bytes]
PAIR = tuple[bytes, bytes]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def update_token_freqs(
    token_freqs: dict[TOKEN, int],
    text_segment: str,
):
    matches = re.finditer(PAT, text_segment)
    for m in matches:
        token = m.group()
        token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))
        token_freqs[token_bytes] += 1 


def process_chunk(args):
    input_path, start, end, special_tokens, chunk_id = args
    # Read chunk
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    # Decode
    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    
    # Logic from original train_bpe
    special_pat = "|".join(re.escape(st) for st in special_tokens)
    segments = re.split(special_pat, chunk_str)
    
    token_freqs = defaultdict(int)
    # Use position=chunk_id+1 so 0 is left for the main bar
    for seg in segments:
        update_token_freqs(token_freqs, seg)
    
    return token_freqs

def count_token_freqs_parallel(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 4,
):
    with open(input_path, 'rb') as f:
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
        boundaries = find_chunk_boundaries(f, num_processes, split_token) 
    
    token_freqs = defaultdict(int)

    ## Initiate token_freqs
    
    tasks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        tasks.append((input_path, start, end, special_tokens, i))
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(process_chunk, tasks), total=len(tasks), desc="Token Freqs (Chunks)", position=0))
        
        for local_freqs in results:
            for token, count in local_freqs.items():
                token_freqs[token] += count

    return token_freqs

def get_pairs(
    token: TOKEN
) -> list[PAIR]:
    if len(token) < 2:
        return []
    return [(token[i], token[i+1]) for i in range(len(token)-1)]

def update_pair_freqs(
    pair_freqs: dict[PAIR, int],
    token_freqs: dict[TOKEN, int],
):
    for token, freq in token_freqs.items():
        if len(token) < 2:
            continue
        pairs = get_pairs(token)
        for p in pairs:
            pair_freqs[p] += freq

def update_pair2idx(
    pair_to_idx: dict[PAIR, set[TOKEN]],
    token_freqs: dict[TOKEN, int],
    token_to_idx: dict[TOKEN, int],
):
    for token, freq in token_freqs.items():
        if len(token) < 2:
            continue
        pairs = get_pairs(token)
        for p in pairs:
            pair_to_idx[p].add(token_to_idx[token])


def get_most_frequent_pair(
    pair_freqs: dict[PAIR, int],
) -> PAIR:
    return max(pair_freqs.keys(), key=lambda k: (pair_freqs[k], k))

def update_vocab(
    vocab: dict[int, bytes],
    new_id: int,
    best_pair: PAIR,
):
    new_vocab = best_pair[0] + best_pair[1]
    vocab[new_id] = new_vocab

def update_vocab_inverse(
    vocab_inverse: dict[bytes, int],
    new_id: int,
    best_pair: PAIR,
):
    new_vocab = best_pair[0] + best_pair[1]
    vocab_inverse[new_vocab] = new_id

def update_merges(
    merges: list[PAIR],
    best_pair: PAIR,
):
    merges.append(best_pair)

def update_all(
    pair_to_idx: dict[PAIR, set[int]],
    pair_freqs: dict[PAIR, int],
    idx_freqs: dict[int, int],
    idx_to_token: dict[int, TOKEN],
    best_pair: PAIR,
):
    affected_idxs = list(pair_to_idx[best_pair])
    merged_bytes = best_pair[0] + best_pair[1]

    for idx in affected_idxs:
        # get new token
        i=0
        token = idx_to_token[idx]
        new_token = []
        while(i<(len(token))):
            if (i < len(token) - 1) and (token[i] == best_pair[0]) and (token[i+1] == best_pair[1]):
                new_token.append(merged_bytes)
                i = i + 2
            else:
                new_token.append(token[i])
                i = i + 1
        new_token = tuple(new_token)


        ## update pair_to_idx
        new_pairs = get_pairs(new_token)
        affected_pairs = get_pairs(token)
        for p in set(new_pairs)-set(affected_pairs):
            pair_to_idx[p].add(idx)
        for p in set(affected_pairs)-set(new_pairs):
            pair_to_idx[p].discard(idx)


        ## update pair_freqs
        for pair in affected_pairs:
            pair_freqs[pair] -= idx_freqs[idx]
        for pair in new_pairs:
            pair_freqs[pair] += idx_freqs[idx]


        # ## update token_freqs
        # origin_freq = token_freqs.pop(token)
        # token_freqs[new_token] = origin_freq

        ## update idx_to_token
        idx_to_token[idx] = new_token
        

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4
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

    print("Update Token Freq (Parallel)...")
    t0 = time.time()

    token_freqs = count_token_freqs_parallel(
        input_path,
        special_tokens,
        num_processes,
    )
                
    idx_to_token = {i: k for i, k in enumerate(token_freqs.keys())}
    token_to_idx = {k: i for i, k in enumerate(token_freqs.keys())}
    idx_freqs = {token_to_idx[token]: freq for token, freq in token_freqs.items()}
    print(f"Update Token Freq took {time.time() - t0:.2f}s")


    ## Initiate pair_freqs, pair_to_token, and token_to_pair
    print("Update Pair Freq...")
    t0 = time.time()
    pair_freqs = defaultdict(int)
    update_pair_freqs(pair_freqs, token_freqs)
    print(f"Update Pair Freq took {time.time() - t0:.2f}s")

    print("Update Pair to idx...")
    t0 = time.time()
    pair_to_idx = defaultdict(set)
    update_pair2idx(pair_to_idx, token_freqs, token_to_idx)
    print(f"Update Pair to Token took {time.time() - t0:.2f}s")

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
        update_vocab(vocab, new_id, best_pair)

        ## update vocab_inverse
        update_vocab_inverse(vocab_inverse, new_id, best_pair)

        ## update merges
        update_merges(merges, best_pair)

        update_all(pair_to_idx, pair_freqs, idx_freqs, idx_to_token, best_pair)
    print(f"Merging took {time.time() - t0:.2f}s")
    print(f"Total Training took {time.time() - start_total:.2f}s")

    return vocab, merges
    

if __name__ == "__main__":

    data_group = "train"
    # file_name = f"TinyStoriesV2-GPT4-{data_group}.txt"
    file_name = f"owt_{data_group}.txt"
    input_path = f"data/{file_name}"
    vocab_size = 32000
    special_tokens = ['<|endoftext|>']

    

    vocab_path = f"outputs/{file_name.split(".")[0]}-vocab-{vocab_size}.pkl"
    merges_path = f"outputs/{file_name.split(".")[0]}-merge-{vocab_size}.pkl"


    vocab, merges = train_bpe(input_path, vocab_size, special_tokens,num_processes=16)
        
    longest_values = sorted(vocab.values(), key=len, reverse=True)[:30]

    for val in longest_values:
        print(f"Bytes: {val} | Size: {len(val)}")
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)