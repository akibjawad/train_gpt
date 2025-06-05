"""
Download the HellaSwag dataset and evaluate the model on it.
https://github.com/rowanz/hellaswag

Example hellawag test item:
{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 124M:
- eleuther harness reports acc of 28.92% acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

"""

import os
import json
import requests
import tiktoken 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# Download the dataset
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'hellaswag')

def download_file(url, fname:str, chunk_size=1024):
    """
    Download a file from a URL and save it to a local file.
    Args:
        url (str): The URL to download the file from.
        fname (str): The local file path to save the downloaded file.
        chunk_size (int): The size of each chunk to read from the response.
    """

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(fname, 'wb') as f, tqdm(
        desc=f"Downloading {fname}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)
hellaswags = {
    'train': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl',
    'val': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl',
    'test': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl'
}

encoder = tiktoken.get_encoding("gpt2")

def download(split:str):
    """
    Download the HellaSwag dataset split.
    Args:
        split (str): The dataset split to download ('train', 'val', or 'test').
    """
    if split not in hellaswags:
        raise ValueError(f"Invalid split: {split}. Choose from {list(hellaswags.keys())}.")
    
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    fname = os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl')
    if not os.path.exists(fname):
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        print(f"Downloading {hellaswags[split]} to {fname}")
        download_file(hellaswags[split], fname)
    return fname

def render_example(example):
    """
    Given the example, a dictionary, render it as three torch tensors:
    - tokens: tokens of the context + completion of size 4xN, as there are 4 completions
    - mask: 1 in the region of completion, where we evaluate likelihood
    - labels: the index of the correct completion, which we have the highest likelihood
    """

    ctx = example['ctx']
    endings = example['endings']
    label = example['label']
    assert len(endings) == 4, "There should be exactly 4 endings."

    data = {
        "labels": label,
        "ctx_tokens": None,
        "endings_tokens": []
    }

    ctx_tokens = encoder.encode(ctx)
    data['ctx_tokens'] = torch.tensor(ctx_tokens)
    tok_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = encoder.encode(" " + ending) # prepnending a space to match because of some quirks in gpt2 tokenization

        tok_row = ctx_tokens + ending_tokens
        tok_rows.append(tok_row)
        mask_row = [0] * len(ctx_tokens) + [1] * len(ending_tokens)
        mask_rows.append(mask_row)

        data['endings_tokens'].append(torch.tensor(tok_row))
    
    # collate all 4 options into a single tensor
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    """
    Iterate over the examples in the HellaSwag dataset split.
    Args:
        split (str): The dataset split to iterate over ('train', 'val', or 'test').
    Yields:
        dict: The example data.
    """
    fname = download(split)
    with open(fname, 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

#evaluate the model on the dataset on the GPT2 model from huggingface
@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') #using tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    # you can also use model = torch.compile(model)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples('val'):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits for the endings
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all postions of the endings
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)  # Reshape to (4, N)

        # now average the losses over the ending tokens, mask==1 for each row
        shift_mask = (mask[..., 1:]).contiguous()  # mask for the ending tokens
        masked_shift_losses = shift_losses * shift_mask
        # sum the losses for each row and divide by number of 1s in the mask
        sum_losses = masked_shift_losses.sum(dim=1)
        avg_losses = sum_losses / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 endings
        # get the index of the minimum loss
        pred = sum_losses.argmin().item()
        pred_norm = avg_losses.argmin().item()

        # accumulate the statistics
        num_total += 1
        if pred == label:
            num_correct += 1
        if pred_norm == label:
            num_correct_norm += 1   
        print(f'num_total: {num_total}, acc_norm: {num_correct_norm}/{num_total} = {num_correct_norm/num_total:.4f}')

        # debugging to print some examples
        if num_total < 10:
            print('-----')
            print(f'Context:\n {example["ctx"]}')
            print(f'Endings:')
            for i, ending in enumerate(example['endings']):
                print(f'{i}: (loss: {avg_losses[i].item()}) {ending}')
            print(f'predicted: pred_norm: {pred_norm}, actual label: {label}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the HellaSwag dataset with a GPT-2 model.')
    parser.add_argument('-m', '--model_type', type=str, default='gpt2', help='The type to use.')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on.')

    args = parser.parse_args()
    
    evaluate(args.model_type, args.device)
