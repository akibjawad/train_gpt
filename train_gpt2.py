
# -------------------------------------------------------
# Train a GPT-2 model on the FineWeb10B dataset
import torch
import torch.nn.functional as F
import os
import math
# import the model class and configuration
from model import GPT, GPTConfig
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
    print()
    print('####### Printing generated samples #######')
    for i in range(num_samples):
        tokens = x_gen[i, :max_length].tolist() # (32,)
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i} >{decoded}")
    print()

#hellaswag evaluation
def get_most_likely_row(tokens, mask, logits):
    """
    Get the most likely row from the logits based on the mask.
    Args:
        tokens (torch.Tensor): The input tokens.
        mask (torch.Tensor): The mask for the tokens.
        logits (torch.Tensor): The logits from the model.
    Returns:
        int: The index of the most likely row.
    """
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
    
    return pred_norm

def hellaswag_eval(model, device):
    """
    Evaluate the model on the HellaSwag dataset.
    Args:
        model (GPT): The GPT model to evaluate.
        device (str): The device to run the evaluation on ('cuda', 'cpu', etc.).
    """
    from hellaswag import render_example, iterate_examples

    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples('val')):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits for the endings
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            if pred_norm == label:
                num_correct_norm += 1
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)  # sum the total number of examples
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)  # sum the number of correct examples
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0
    if master_process:
        print(f"HellaSwag Accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"step: {step} hellaswag: {acc_norm:.4f}\n")

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
B = 64 # micro batch size: This is the batch size per forward/backward pass on each device (GPU).
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
# make torch compile optional
use_compile = True # set to True to use torch.compile
if use_compile:
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

## a log directory to save training time logs
log_dir = 'log_train'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f: # write mode will clear the file
    pass


## Set up profiller for tracking time and memory usage
from torch.profiler import profile, record_function, ProfilerActivity
profiler_log_dir = 'log_profiler'
os.makedirs(profiler_log_dir, exist_ok=True)
profiler = torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=3), # profile R × (W + U + A) cycles
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_modules=True,
)
# set to 250 to match result from andrej
eval_step = 250 # how often to log the training loss

with profiler:
    for step in range(max_steps):
        last_step = (step == max_steps - 1)
        t0 = time.time()

        if step % eval_step == 0 or last_step: 
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
                    with open(log_file, 'a') as f:
                        f.write(f"step {step}: val {val_loss_accum.item():.4f}\n")
        
        if (step % eval_step == 0 or last_step): #and (not use_compile):
            # set the model to evaluation
            model.eval()
            #use hellaswag evaluation
            hellaswag_eval(model, device)

        if (step % eval_step == 0 or last_step):
            # set the model to evaluation
            model.eval()
            # sample from the model trained so far
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
            with open(log_file, 'a') as f:
                f.write(f"step {step}: train {loss_accum.item():.6f}")
        profiler.step() # step the profiler to record the time and memory usage
if ddp:
    # saving the model
    if master_process:
        save_path = 'models'
        internal_model = model
        if isinstance(model, DDP):
            internal_model = model.module
        if hasattr(internal_model, '_orig_mod'):
            internal_model = internal_model._orig_mod  # get the original model if it was compiled
        # save the model state dict
        torch.save(internal_model.state_dict(), os.path.join(save_path, 'gpt2_fineweb_124M.pt'))
        # save the model config
        with open(os.path.join(save_path, "config.json"), "w") as f:
            f.write(internal_model.config.to_json_string())

        with open(os.path.join(save_path, "ReadMe.txt"), "w") as f:
            f.write(
            "This is a fine-tuned GPT-2 model on the FineWeb10B dataset.\n"
            "To load the model, use `model.load_state_dict(torch.load('gpt2_fineweb_124M.pt'))`."
        )
        
        # model.module.config.to_json_file(os.path.join(save_path, "config.json"))
        print(f"Saving model to {save_path}")
    
    destroy_process_group()

import sys
sys.exit(0)

