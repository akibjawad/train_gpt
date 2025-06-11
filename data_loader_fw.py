import torch
import os
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
        if self.process_rank == 0:
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