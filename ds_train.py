import torch
import torch.nn.functional as F
import deepspeed
import deepspeed.comm as dist

# initializing distributed
deepspeed.init_distributed()

# data loader portion for training with deepspeed
from data_loader_fw import DataLoaderLite
import os

# validation loss calculation
def calculate_val_loss(model_engine, curr_step, val_loader, val_loss_step, device):
    model_engine.eval() # set the model to evaluation mode
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
                _, loss = model_engine(x, y)
            loss = loss/val_loss_step # scale the loss by the number of accumulation steps
            val_loss_accum += loss.detach() # accumulate the loss
        # if we are using DDP, we need to synchronize the gradients across all processes
        # this is done automatically by DDP when we call loss.backward()
        # but we don't want to do this until we have done all the accumulation steps
        # all_reduce across data-parallel groups only
        dist.all_reduce(val_loss_accum, op=deepspeed.comm.ReduceOp.AVG)
        if dist.get_rank() == 0:
            print(f"Validation loss at step {curr_step}: {val_loss_accum.item():.4f}")
            # with open(log_file, 'a') as f:
            #     f.write(f"step {step}: val {val_loss_accum.item():.4f}\n")
        model_engine.train() # set the model back to training mode


#load the deepspeed config file
import json
import yaml
with open('ds_config.yml') as f:
    ds_config = yaml.safe_load(f)

B = ds_config['train_micro_batch_size_per_gpu'] # full batch size
T = ds_config['block_size'] # context length/ block size
grad_accum_steps = ds_config['gradient_accumulation_steps']

# deepspeed config file requires json format
# convert the yaml config to json
with open('ds_config.json', 'w') as f:
    json.dump(ds_config, f, indent=4)

local_rank = int(os.environ.get('LOCAL_RANK', 0))
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)

# seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#data loader
# ddp_rank = dist.get_data_parallel_rank()
# ddp_world_size = dist.get_data_parallel_world_size()

# running without ddp for now, pure piepeline parallelism
ddp_rank = 0
ddp_world_size = 1



train_loader = DataLoaderLite(batch_size=B, block_size=T, split='train', process_rank=ddp_rank, process_world_size=ddp_world_size)
val_loader = DataLoaderLite(batch_size=B, block_size=T, split='val', process_rank=ddp_rank, process_world_size=ddp_world_size)

def tain_iter():
    for _ in range(grad_accum_steps):
        # get the next batch of data
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        # yield the batch for the pipeline module to process
        yield x, y

def val_iter():
    yield train_loader.next_batch()


# set tf32 matmul precision
torch.set_float32_matmul_precision('high')

# model configuration
from ds_model import GPTConfig, EmbeddingModule, LmHeadModule, Block
from deepspeed.pipe import PipelineModule, LayerSpec
model_config = (GPTConfig(vocab_size=50304))

# gather layers for the pipeline
layers = []
layers.append(LayerSpec(EmbeddingModule, model_config))

for _ in range(model_config.n_layer):
    layers.append(LayerSpec(Block, model_config))
layers.append(LayerSpec(LmHeadModule, model_config))



# define cross entropy loss function
# this is used in the forward method of the pipeline module
def cross_entropy_loss_fn(logits, labels):
    """
    Custom cross entropy loss function for the pipeline module.
    Args:
        logits (torch.Tensor): The output logits from the model.
        labels (torch.Tensor): The ground truth labels.
    Returns:
        torch.Tensor: The computed cross entropy loss.
    """
    # reshape logits and labels to match the expected shape
    # (B, T, vocab_size) -> (B*T, vocab_size)
    logits = logits.view(-1, model_config.vocab_size)
    # (B, T) -> (B*T)
    labels = labels.view(-1)
    # calculate the loss
    loss = F.cross_entropy(logits, labels)
    return loss

def configure_pipeline_optimizers(pipeline_model, weight_decay=0.1, learning_rate=6e-4, device='cuda'):
    import inspect
    from torch.optim import AdamW

    param_dict = {pn: p for pn, p in pipeline_model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    fused_available = 'fused' in inspect.signature(AdamW).parameters
    use_fused = fused_available and 'cuda' in device

    optimizer = AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused
    )

    return optimizer

# lr scheduler
from torch.optim.lr_scheduler import _LRScheduler
import math

class SimpleGetLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # epoch is 1 indexed but step is calculated based on 0 indexed
        step = self.last_epoch - 1
        print(f"[LRScheduler] Step = {step}")
        # 1) linear warmup for warmup_steps
        if step < self.warmup_steps:
            return [self.max_lr * (step+1) / self.warmup_steps] # step is 0-indexed
        # 2) if step > lr_decay_iters, return min_lr
        if step > self.max_steps:
            return [self.min_lr]
        # originally in the gpt3 paper, before they start decaying the used 0.1 * max_lr
        # 3) decay the learning rate using cosine schedule
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio)) # coeff starting from 1 to 0
        return [self.min_lr + coeff * (self.max_lr - self.min_lr)]


# create the pipeline model
model = PipelineModule(layers=layers,
                       loss_fn=cross_entropy_loss_fn,
                       num_stages=1,
                       partition_method='parameters',
                       )
# deepspped internally moves the model to the correct device
# so we don't need to call model.to(device) here

# configure the optimizer
optimizer = configure_pipeline_optimizers(model, weight_decay=0.1,learning_rate=6e-4,device=device)
max_lr = 6e-4 #following gpt3 paper
min_lr = max_lr * 0.1 # 10% of max_lr
warmup_steps = 715 # warmup lr until 375M tokens, hence warm up until 375M/2**19
max_steps = 19073 # per step tokens processed is 524288 (2**19), so 19073 steps will be 524288 tokens processed

lr_scheduler = SimpleGetLRScheduler(
    optimizer=optimizer,
    warmup_steps=warmup_steps,  # number of warmup steps
    max_steps=max_steps,  # total number of training steps
    max_lr=max_lr,  # maximum learning rate
    min_lr=min_lr,  # minimum learning rate
)
# lr scheduler is configured in the deepspeed config file following the previous code
#### for manual configuration we could've used the following code
# custom_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(custom_optimizer, get_lr)

# deepspeed training engine
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    lr_scheduler=lr_scheduler,
    config_params=ds_config
    # mpu=None,  # not using model parallelism
)

model_engine.train()  # set the model to training mode
import time
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # get the next batch of data
    # use iterators to yield batches for the pipeline module

    # forward pass & backward pass & gradient accumulation & optimizer step
    # in pipeline parallelism, we need to call train_batch
    model_engine.train_batch(data_iter=tain_iter()),

    # backward pass
    loss = model_engine.loss

    # step the optimizer
    # model_engine.step()

    # print training loss every 100 steps
    if step % 100 == 0 and local_rank == 0:
        print(f"Step {step}: Training loss: {loss.item():.4f}")

    # validate every 1000 steps
    # if step % 100 == 0 and local_rank == 0:
        # calculate_val_loss(model_engine, step, val_loader, val_loss_step=20, device=device)

    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_processed = (B * T) * grad_accum_steps * ddp_world_size
    tokens_per_second = (tokens_processed / dt)*1000 # tokens per second, because dt is in milliseconds

    if local_rank == 0:
        curr_loss = loss.item()
        curr_lr = model_engine.get_lr()[0]
        # you cannot get the global gradient norm directly from the model engine
        # grad_norm = model_engine.get_gradient_norm()
        print(f"Step {step} | loss: {loss.item():.6f} | lr:{curr_lr:0.4e} | dt {dt:.2f}ms | {tokens_per_second:.2f} tokens/sec")
