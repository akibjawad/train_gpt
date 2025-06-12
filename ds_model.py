# keeping other classes in the model.py same as before

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import json

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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (math.sqrt(k.size(-1)))) # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # (B,nh,T,T) -> (B,nh,T,T)
        # att = F.softmax(att, dim=-1)
        # y = (att @ v) # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)

        # use flassh attention instead of the regular attention calculation
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        # reassmble the heads back into the embedding dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B,nh,T,hs) -> (B,T,nh*hs) -> (B,T,C)

        y = self.c_proj(y) # (B,T,C) -> (B,T,C)
        return y
    


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.apply(self._init_weights)  # initialize the weights of the block

    def forward(self, x):
        # from the gpt2 paper: 
        # "We scale the weights of the residual layers at initialization by 1/sqrt(N) where N is the number of residual layers in the block"
        # to implement this follw the flag NANOGPT_SCALE in c_proj (last layer of each block)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def _init_weights(self, module):
        # initialize the weights of the module
        if isinstance(module, nn.Linear):
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

# defining the final layer norm and lm_head
class LmHeadModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)  # initialize the weights of the lm_head

    def forward(self, x):
        x = self.ln_f(x)
        # deepspeed will process y and calculate the loss
        # with the provided function as loss_fn
        # hence no need to return loss here
        loss = None 
        logits = self.lm_head(x)
        return logits
    
    def _init_weights(self, module):
        # initialize the weights of the lm_head
        # layer norm is initialized by default from pytorch to be mean=0, std=1
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            # initialize the linear layer weights (lm_head)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

# defining the embedding layers
class EmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.apply(self._init_weights)  # initialize the weights of the embedding layers

    def forward(self, x):
        B, T = x.shape # B is batch size, T is sequence length
        pos = torch.arange(0, T, device=x.device, dtype=torch.long) # create position tensor
        tok_emb = self.wte(x) # token embeddings #(B, T, n_embd)
        pos_emb = self.wpe(pos) # position embeddings #(T, n_embd)
        x = tok_emb + pos_emb # add position embeddings to token embeddings, # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        return x
    
    def _init_weights(self, module):
        # initialize the weights of the embedding layers
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=4, sort_keys=True) + "\n"


##### Process loss function and the configure optimizer later in the pipeline module
# generation will not be implemented in the pipeline module, it will be done after the training
