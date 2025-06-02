from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

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
    
    def configure_optimizers(self, weight_decay=0.1, learning_rate=6e-4, device='cpu'):
        # configure the optimizer 
        # find parameters that should decay weights (requires_grad and dim>=2) and those should not decay weights (not requires_grad)
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create a optim_groups. Any parameter that is 2D or more will be in the decay group, else in the no decay group
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2] # bias and layernorm, etc
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum([p.numel() for p in decay_params])
        num_no_decay_params = sum([p.numel() for p in no_decay_params])
        if master_process:
            print(f'num decay param tensors: {len(decay_params)}, with num decay parameters: {num_decay_params}')
            print(f'num no decay param tensors: {len(no_decay_params)}, with num no decay parameters: {num_no_decay_params}')
            
        # check if fused adamw is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"Using fused adamw: {use_fused}")
        # use AdamW with weight decay
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer

# -------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    """Load tokens from a shard file and return them as a tensor."""
    # npt = np.load(filename)
    npt = np.memmap(filename, dtype=np.uint16, mode='r')
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, batch_size, block_size, split, process_rank=0, process_world_size=1):
        self.B = batch_size
        self.T = block_size
        self.process_rank = process_rank
        self.num_processes = process_world_size

        assert split in ['train', 'val', 'test'], f"Invalid split: {split}, must be one of ['train', 'val', 'test']"
        
        # get the shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [f for f in shards if split in f]
        shards = sorted(shards) # sort the shards to ensure consistent order
        shards = [os.path.join(data_root, f) for f in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split} in {data_root}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}:")
        # state
        self.reset() # reset the dataloader to the beginning of the first shard
    
    def reset(self):
        """Reset the dataloader to the beginning of the current shard."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # self.current_position = 0 # position in 0 for single gpu traning
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        # you can move those to gpu later when required
        buff = self.tokens[self.current_position : self.current_position + self.B * self.T +1] # (B*T+1,)
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)

        # advance the position in the tensor
        self.current_position += self.B * self.T * self.num_processes

        # if next batch crosses the length of current shard, load the next shard
        if self.current_position + (self.B * self.T * self.num_processes + 1) > self.tokens.size(0):
            # self.current_position = 0
            # for multi-gpu running
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard]) # load the next shard
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
# -------------------------------------------------------
# Sample from a GPT-2 model

#### Doing sampling on the model
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
    sample_rng.manual_seed(42+ddp_rank)  # set the seed for reproducibility
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
        print(f"rank {ddp_rank} sample {i} >{decoded}")

# --------- running multi GPU training loop --------------
import os
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# setup the DDP(Distributed Data Parallel) process group
# torchrun command sets the env varraible RANK, LOCAL_RANK, WORLD_SIZE
# with torchrun command you will create multiple processes, all the processes will be running the same script
# Hence all code below this line will be multi-threaded/multi-process
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp process?
if ddp:
    # print("Running in DDP mode")
    assert torch.cuda.is_available(), "CUDA is not available, but DDP requires it"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # print(f"ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}")
    # set the device to the local rank
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # This process will do logging and checkpointing etc
else:
    print("Running in single GPU mode")
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # set the device appropriately
    if torch.cuda.is_available():
        print("CUDA is available, using GPU")
        device = 'cuda'
    elif torch.backends.mps.is_available():
        print("MPS is available, using GPU")
        device = 'mps'
    else:
        print("No GPU available, using CPU")
        device = 'cpu'
    print(f"Using device {device} for training")


# #testing on cpu
# device = 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation: 
# you don't have the resources to train in 1 large batch of total_batch_size
# so you split the batch into B*T and accumulate the gradients with doing forward and backward pass and not updating the weights
# once you do accumulation_steps passes, you update the weights

# total_batch_size = 12*1024 # experiment matching previous result with 12 batch size
# B = 4 # experiment matching previous result with 12 batch size

total_batch_size = 524288 # 2**19, ~0.5M tokens (from the paper gpt3) # number of tokens processed accross all GPUs per step
B = 8 # micro batch size: This is the batch size per forward/backward pass on each device (GPU).
T = 1024 # context length/ block size
# adjusting for the multi-process training
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size {total_batch_size} is not divisible by B*T*ddp_word_size {B*T*ddp_world_size}"

# If your batch doesn't fit in memory, you can accumulate gradients over multiple steps before updating weights. This simulates a larger batch size.
accumulation_steps = total_batch_size // (B * T * ddp_world_size)
# all 8 process will print this line, instead make sure only the master process prints it
if master_process:
    print(f"total_batch_size: {total_batch_size}, accumulation_steps: {accumulation_steps}")

# test multiple GPUs
print(f"I am process {ddp_rank} of {ddp_world_size} processes running on {device}")

# import sys; sys.exit(0) # for the test running, 
# done running on 2 gpus

# initialize the dataloader
# each process will create its own dataloader
train_loader = DataLoaderLite(batch_size=B, block_size=T, split='train', process_rank=ddp_rank, process_world_size=ddp_world_size)
val_loader = DataLoaderLite(batch_size=B, block_size=T, split='val', process_rank=ddp_rank, process_world_size=ddp_world_size)


# set tf32
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
# print("model weight loading successful, you didn't crash")
# each process will create its own model
model = GPT(GPTConfig(vocab_size=50304)) # simply overriding 50257 to 50304 so that pytorch handles the vocab size better
# model = GPT(GPTConfig())
print(f'creating random model on {device}')
model.to(device)
# disabling compile for now
model = torch.compile(model)
# wrap the model with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    # since we are using DDP, it will automatically synchronize the gradients across all processes
    # Once the loss.backward() is done, it will call allreduce(), the gradients are averaged across all processes
    # and the averaged gradients will be provided to each process
    # to update the weights
# create a raw model to get the optimizer
raw_model = model.module if ddp else model
    

# learning rate schedule, linear warmup and cosine decay to a minimum
max_lr = 6e-4 #following gpt3 paper
min_lr = max_lr * 0.1 # 10% of max_lr

# warmup_steps = 10
# max_steps = 50

warmup_steps = 715 # warmup lr until 375M tokens, hence warm up until 375M/2**19
max_steps = 19073 # per step tokens processed is 524288 (2**19), so 19073 steps will be 524288 tokens processed

def get_lr(step):
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps # step is 0-indexed
    # 2) if step > lr_decay_iters, return min_lr
    if step > max_steps:
        return min_lr
    # originally in the gpt3 paper, before they start decaying the used 0.1 * max_lr
    # 3) decay the learning rate using cosine schedule
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio)) # coeff starting from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)

# training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# hyperparameters tuning
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.99,0.995), eps=1e-8)

## Set up profiller for tracking time and memory usage
from torch.profiler import profile, record_function, ProfilerActivity

profiler = torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=3), # profile R × (W + U + A) cycles
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_modules=True,
)

with profiler:
    for step in range(max_steps):
        if step >= 30:
            break # for testing purposes, break after 30 steps
        t0 = time.time()
        if step % 10 == 0:
            model.eval() # set the model to evaluation mode
            # evaluate the model on the validation set
            val_loader.reset() # reset the validation loader to the beginning of the first shard
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_step = 20 # how many steps to accumulate the validation loss
                for _ in range(val_loss_step):
                    x, y = val_loader.next_batch()
                    x = x.to(device)
                    y = y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        # parameters are still in float32 but logits are in bfloat16
                        # only the matrix multiplications are in bfloat16
                        # softmax, loss, and other operations are in float32
                        logits, loss = model(x, y)
                    loss = loss/val_loss_step # scale the loss by the number of accumulation steps
                    val_loss_accum += loss.detach() # accumulate the loss
                if ddp:
                    # if we are using DDP, we need to synchronize the gradients across all processes
                    # this is done automatically by DDP when we call loss.backward()
                    # but we don't want to do this until we have done all the accumulation steps
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG) # sum the loss across all processes
                if master_process:
                    print(f"Validation loss at step {step}: {val_loss_accum.item():.4f}")
        
        if step > 0 and step % 10 == 0:
            # set the model to evaluation
            model.eval()
            do_sampling(model,device)

        model.train() # set the model to training mode
        # training loop
        optimizer.zero_grad() # zero the gradients, set_to_none=True is more memory efficient
        loss_accum = 0.0
        for micro_steps in range(accumulation_steps):
            # getting a batch of data
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)
            # print(f"Input shape: {x.shape}, Output shape: {y.shape}")
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # parameters are still in float32 but logits are in bfloat16
                # only the matrix multiplications are in bfloat16
                # softmax, loss, and other operations are in float32
                logits, loss = model(x, y) # (B, T, vocab_size), (B, T)
            loss = loss / accumulation_steps # scale the loss by the number of accumulation steps
            loss_accum += loss.detach() # accumulate the loss
            if ddp:
                # if we are using DDP, we need to synchronize the gradients across all processes
                # this is done automatically by DDP when we call loss.backward()
                # but we don't want to do this until we have done all the accumulation steps
                # that's why we call the loss.backward() at the last micro step
                # we can do this with the ddp.no_sync() context manager
                # we can also directly set the require_backward_grad_sync flag to False
                model.require_backward_grad_sync = (micro_steps == accumulation_steps - 1)
                # kind of hacky, but currently it works
            loss.backward() # set gradients
        
        if ddp:
            # loss_accum is not part of the gradient graph,
            # so each process has it's own loss_accum
            # we don't want to print the local loss_accum
            # we want to prin the average loss_accum across all processes
            # because that is the correct loss for this step
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # sum the loss across all processes
            

        # clip the gradients so model doesn't get a shock with large gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # detect and set the learning rate before applying the update
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step() # update the weights
        torch.cuda.synchronize() # wait till GPU is done with all the works
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_processed = (B * T) * accumulation_steps * ddp_world_size
        tokens_per_second = (tokens_processed / dt)*1000 # tokens per second, because dt is in milliseconds
        # tokens_per_second = (train_loader.B * train_loader.T) / dt
        if master_process:
            print(f"Step {step} | loss: {loss_accum.item():.6f} | lr:{lr:0.4e} | norm {norm:0.4f} | dt {dt:.2f}ms | {tokens_per_second:.2f} tokens/sec")
        profiler.step() # step the profiler to record the time and memory usage
if ddp:
    destroy_process_group()

import sys
sys.exit(0)

