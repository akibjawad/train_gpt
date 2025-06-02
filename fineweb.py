"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the FineWeb-Edu dataset and save the shards in disk.
run as python fineweb.py
"""

import os   
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# ---------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) #100000000 # 100M tokens per shard
# ---------------

#create local directory
DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

#download dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

#initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"] # end of text token

def tokenize_function(doc):
    # tokenize the document and add end of text token
    # returns a list of tokens as numpy array
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 65536).all(), "Some token IDs are out of range"
    tokens_np = tokens_np.astype(np.uint16)  # convert to uint16
    return tokens_np

def write_datafile(filename, data):
    # write the data (numpy array of uint16) to a binary file
    with open(filename, "wb") as f:
        f.write(data.tobytes())

# tokenize all documents in parallel and write output shards
# each shard is set to shard_size 100M tokens
# last shard may be smaller

nprocs = max(1, os.cpu_count()-2)  # use 1/2 cores for other tasks
progress_bar = None

with mp.Pool(processes=nprocs) as pool:
    shard_index = 0
    # preallocate a buffer to hold the current shard
    all_tokens = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    for tokens in pool.imap(tokenize_function, fw, chunksize=16):
        # check if we need to write a new shard
        if token_count + len(tokens) < shard_size:
            # add tokens to the current shard
            all_tokens[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"Processing shard {shard_index:06d}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard to disk and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            # split the dataset into whatever can fit into current shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)

            all_tokens[token_count:token_count + remainder] = tokens[:remainder]
            # write to file
            write_datafile(filename, all_tokens)
            shard_index += 1
            #reset progress bar
            progress_bar = None

            # create the next shard
            all_tokens = np.empty((shard_size,), dtype=np.uint16)
            token_count = len(tokens) - remainder
    
    # if any tokens are left in the buffer, write them to disk
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens[:token_count])


