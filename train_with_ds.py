import torch
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
        deepspeed.comm.all_reduce(val_loss_accum, op=deepspeed.comm.ReduceOp.AVG)
        if deepspeed.comm.get_rank() == 0:
            print(f"Validation loss at step {curr_step}: {val_loss_accum.item():.4f}")
            # with open(log_file, 'a') as f:
            #     f.write(f"step {step}: val {val_loss_accum.item():.4f}\n")
        model_engine.train() # set the model back to training mode


#load the deepspeed config file
import json
import yaml
with open('ds_config.yml') as f:
    ds_config = yaml.safe_load(f)

B = ds_config['train_micro_batch_size_per_gpu'] # micro batch size
T = ds_config['block_size'] # context length/ block size

# deepspeed config file requires json format
# convert the yaml config to json
# with open('ds_config.json', 'w') as f:
#     json.dump(ds_config, f, indent=4)


import math
import deepspeed
import time
from model_pipeline import GPTPipelineModule, GPTConfig

local_rank = int(os.environ.get('LOCAL_RANK', 0))
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)

# seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#data loader
ddp_rank = deepspeed.runtime.comm.get_data_parallel_rank()
ddp_world_size = deepspeed.runtime.comm.get_data_parallel_world_size()

train_loader = DataLoaderLite(batch_size=B, block_size=T, split='train', process_rank=ddp_rank, process_world_size=ddp_world_size)
val_loader = DataLoaderLite(batch_size=B, block_size=T, split='val', process_rank=ddp_rank, process_world_size=ddp_world_size)

# learning rate scheduler defined in the deepspeed config file

# set tf32 matmul precision
torch.set_float32_matmul_precision('high')

# model configuration
model = GPTPipelineModule(GPTConfig(vocab_size=50304))
model.to(device)  # move the model to the current device
custom_optimizer = model.configure_optimizers()
# lr scheduler is configured in the deepspeed config file following the previous code
#### for manual configuration we could've used the following code
# custom_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(custom_optimizer, get_lr)

# deepspeed training engine
model_engine, optimizer, _, = deepspeed.initialize(
    model=model,
    optimizer=custom_optimizer,
    model_parameters=model.parameters(),
    config_params=ds_config
    # mpu=None,  # not using model parallelism
)

# training loop
max_steps = 19073
model_engine.train()  # set the model to training mode

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # get the next batch of data
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)

    # forward pass
    logits, loss = model_engine(x, y)

    # backward pass
    model_engine.backward(loss)

    # step the optimizer
    model_engine.step()

    # print training loss every 100 steps
    if step % 100 == 0 and local_rank == 0:
        print(f"Step {step}: Training loss: {loss.item():.4f}")

    # validate every 1000 steps
    if step % 100 == 0 and local_rank == 0:
        calculate_val_loss(model_engine, step, val_loader, val_loss_step=20, device=device)

    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_processed = (B * T) * ds_config['gradient_accumulation_steps'] * ddp_world_size
    tokens_per_second = (tokens_processed / dt)*1000 # tokens per second, because dt is in milliseconds

    if local_rank == 0:
        curr_loss = loss.item()
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step} | loss: {loss.item():.6f} | lr:{curr_lr:0.4e} | dt {dt:.2f}ms | {tokens_per_second:.2f} tokens/sec")
