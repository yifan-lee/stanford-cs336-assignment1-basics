import regex as re

from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



test_string = "abere erererere<|endoftext|>When and where is not as important as who and what. Hi, I am the the Ivan.<|endoftext|> aaa"
input_path: str = ""
vocab_size: int = 260
special_tokens: list[str] = ['<|endoftext|>']

vocab: dict[int, bytes] = {i:bytes([i]) for i in range(256)}
vocab_inverse: dict[int, bytes] = {v:k for k,v in vocab.items()}
merges: list[tuple[bytes, bytes]] = []


new_id = 256


special_pat = "|".join(re.escape(st) for st in special_tokens)
segments = re.split(special_pat, test_string)
text_segment = segments[0]
matches = re.finditer(PAT, text_segment)
word_freqs = defaultdict(int)
pair_freqs = defaultdict(int)
pair_to_tokens = defaultdict(set)
text_segment



for match in matches:
    token = match.group()
    token_bytes = token.encode("utf-8")
    word_freqs[token_bytes] += 1
    if len(token_bytes)>1:
        for i in range(1, len(token_bytes)):
            pair = (bytes([token_bytes[i-1]]), bytes([token_bytes[i]]))
            pair_freqs[pair] += 1
            pair_to_tokens[pair].add(token_bytes)
pair_freqs


### find most frequent pair
best_pair = max(pair_freqs.keys(), key=lambda k: (pair_freqs[k], k))
vocab[new_id] = best_pair
vocab_inverse[best_pair] = new_id
merges.append(best_pair)
best_pair


affected_tokens = pair_to_tokens[best_pair]
merged_bytes = best_pair[0] + best_pair[1]

token = (next(iter(affected_tokens)))
token


new_neighbors = []
i=0
duplications_count = 0
new_token = []

while(i<(len(token)-1)):
    if (bytes([token[i]]) == best_pair[0]) and (bytes([token[i+1]]) == best_pair[1]):
        print('yes')
        new_token.append(merged_bytes)
        i = i + 2
        duplications_count += 1
    else:
        new_token.append(bytes([token[i]]))
        i = i + 1
print(new_token)