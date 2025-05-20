from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):
    """ Multi-head masked (combined in to a single head of size n_embed for simpler computation) self-attention module """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads but in a batched way
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # for regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embd
        # not really a bias, it's just a mask, following naming pattern of OpenAI/HuggingFace
        # inital 1, 1 are for batch and nh dimensions, helps to understand the broadcasting
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape # B is batch size, T is sequence length, C is number of channels

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh in "number of heads", hs is "head size", and C is "number of channels" = nh * hs
        # for example, in GPT2 (124M params), nh=12, hs=64, C=768

        # get qkv combined
        qkv = self.c_attn(x) # x(B,T,C) -> qkv(B,T,3*C)
        # qkv = [ [ QQQQQQ | KKKKKK | VVVVVV ],
        #           [ QQQQQQ | KKKKKK | VVVVVV ],
        #           ...
        #           [ QQQQQQ | KKKKKK | VVVVVV ], 
        #       ]

        # split into q, k, v
        q,k,v = qkv.split(self.n_embed, dim=2) # split into 3 tensors of size (B,T,C)
        # q = [ [QQQQQQ], [QQQQQQ], ... ]
        # k = [ [KKKKKK], [KKKKKK], ... ]
        # v = [ [VVVVVV], [VVVVVV], ... ]   

        # C dimension is broken into n_heads self-attention heads where each head has size hs = C / n_heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> after transpose (B,nh,T,hs) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs)
        # this way n_head is like a batch dimension, making sure n_heads are computed in parallel


        # attention mask is the (B, nh, T, T) tensor
        att = (q @ k.transpose(-2, -1)) * (1.0 / (math.sqrt(k.size(-1)))) # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B,nh,T,T) -> (B,nh,T,T)
        att = F.softmax(att, dim=-1)
        y = (att @ v) # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B,nh,T,hs) -> (B,T,nh*hs) -> (B,T,C)
        # reassble the heads back into the embedding dimension

        y = self.c_proj(y) # (B,T,C) -> (B,T,C)
        return y
    


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # from the gpt2 paper: 
        # "We scale the weights of the residual layers at initialization by 1/sqrt(N) where N is the number of residual layers in the block"
        # to implement this follw the flag NANOGPT_SCALE in c_proj (last layer of each block)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # same data is used for both lm_head and wte
        self.transformer.wte.weight = self.lm_head.weight

        # initialize the weights to follow the gpt2 implementation done by openai
        # apply the init_weights function to all modules in the model
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # if the module has the NANOGPT_SCALE_INIT attribute, use it to scale the weights
                # 2 comes from 2 types of layers (mlp and attn)
                # this is where we follow the quote from the gpt2 paper
                # "We scale the weights of the residual layers at initialization by 1/sqrt(N) where N is the number of residual layers in the block"
                # to implement this follow the flag NANOGPT_SCALE in c_proj (last layer of each block)

                std *= (2*self.config.n_layer) ** -0.5 # 2 comes from 2 types of layers (mlp and attn)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # layerNorm is not initialized explicitly, it is initialized by default from pytorch to be mean=0, std=1
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # elif isinstance(module, CausalSelfAttention):
        #     # initialize the attention weights
        #     module.c_attn.weight.data.normal_(mean=0.0, std=0.02)
        #     module.c_proj.weight.data.normal_(mean=0.0, std=0.02)


    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pretrained GPT-2 124M model weights from huggingface"""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], f"Model type {model_type} not supported"
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT: {model_type}")

        # n_layer, n_head, n_embd are determined by the model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        #create a from-sratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] # discard the bias of the attention layer

        # get the pretrained model from huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard the bias of the attention layer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard the bias of the attention layer
        # original model has some of the weights transposed, it was written in tensorflow
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # hence we need to transpose the weights
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for Conv1D weights, transpose the weights
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # simply copy weights from model_hf to model
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size}, but input sequence length is {T}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        
        # add the token and position embeddings
        # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # forward through the final layer norm
        x = self.transformer.ln_f(x)

        # get the logits
        loss = None
        logits = self.lm_head(x)
        # (B, T, n_embd) -> (B, T, vocab_size)
        if targets is not None:
            # calculate the loss
            # (B, T, vocab_size) -> (B*T, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            # (B, T) -> (B*T)
            targets = targets.view(-1)
            # calculate the loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor of input tokens
        for _ in range(max_new_tokens):
            # get the logits
            logits, _ = self(idx)
            # get the last token
            logits = logits[:, -1, :]
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the new token to the input
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, datafile, batch_size, block_size):
        with open(datafile, 'r') as f:
            data = f.read()
            f.close()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.B = batch_size
        self.T = block_size
        self.num_batches = len(tokens) // (batch_size * block_size)

        # state
        self.current_position = 0
    
    def next_batch(self):
        # you can move those to gpu later when required
        buff = self.tokens[self.current_position : self.current_position + self.B * self.T +1] # (B*T+1,)
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)
        self.current_position += self.B * self.T

        if self.current_position + (self.B * self.T + 1) > self.tokens.size(0):
            self.current_position = 0
        return x, y
    
# -------------------------------------------------------
# Test the model


if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = 'cuda'
elif torch.backends.mps.is_available():
    print("MPS is available, using GPU")
    device = 'mps'
else:
    print("No GPU available, using CPU")
    device = 'cpu'

# #testing on cpu
# device = 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# initialize the dataloader
train_loader = DataLoaderLite('tiny_shakespeare.txt', batch_size=4, block_size=32)


# model = GPT.from_pretrained('gpt2')
# print("model weight loading successful, you didn't crash")

model = GPT(GPTConfig())
print('creating random model')
model.to(device)

# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    # getting a batch of data
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    # print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    optimizer.zero_grad()
    logits, loss = model(x, y) # (B, T, vocab_size), (B, T)
    loss.backward() # set gradients
    optimizer.step() # update the weights
    print(f"Step {i+1} done loss: {loss.item()}")

import sys
sys.exit(0)

#### Doing sampling on the model

max_return_sequences = 5
max_length = 30

model.eval()
model.to(device)


tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(max_return_sequences,1) # (5,8)
x = tokens.to(device)
print(f"Input tokens: {tokens}")

# generate new tokens: currently x is (B=5, T=8)
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # forward the model to get the logits

    with torch.no_grad():
        logits = model(x) # (B=5, T=8, vocab_size=50257)
        # get the logits at the last token
        logits = logits[:, -1, :] # (B=5, vocab_size=50257)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)

        # do a top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (B=5, k=50), topk_indices becomes (B=5, k=50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) #(B=5, 1)
        # get the corresponding index
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        # append the new token to the input
        x = torch.cat((x, xcol), dim=1)
# print the generated tokens
for i in range(max_return_sequences):
    tokens = x[i, :max_length:].tolist() # (30,)
    decoded = enc.decode(tokens)
    print(f">{decoded}")