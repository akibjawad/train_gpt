import torch
import tiktoken
import torch.nn.functional as F

def do_sampling(model, device):
    """Do sampling on the model and print the generated text."""
    num_samples = 4
    max_length = 32
    # encode the input text
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1) # (4,8) 
    x_gen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)  # set the seed for reproducibility
    while x_gen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, loss = model(x_gen) # (B, T, vocab_size)
            #take the logits at the last token
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do a top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, k)
            # select a token from the top-k probabilities
            # multonomial does not demand the probabilities to be normalized, so we can use topk_probs directly
            ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) #(B, 1)
            # get the corresponding index
            xcol = torch.gather(topk_indices, dim=-1, index=ix) # (B, 1)
            # append the new token to the input
            x_gen = torch.cat((x_gen, xcol), dim=1)
    # print the generated tokens
    for i in range(num_samples):
        tokens = x_gen[i, :max_length].tolist() # (32,)
        decoded = enc.decode(tokens)
        print(f"sample {i} >{decoded}")


from model import GPT, GPTConfig
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#model save path
model_save_path = "models"
# load config
with open(os.path.join(model_save_path, 'config.json'), "r") as f:
    config = json.load(f)
    gpt_config = GPTConfig(**config)

# load model
model = GPT(gpt_config)
# load state dict
state_dict = torch.load(os.path.join(model_save_path, 'gpt2_fineweb_124M.pt'))
# print(state_dict.keys())  # check the keys in the state dict
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # set the model to evaluation mode
do_sampling(model, device)  # run the sampling
