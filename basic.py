import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetv1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.tokenizer= tokenizer
        self.input_ids= []
        self.target_ids= []

        token_ids= tokenizer.encode(txt)
        
        # sliding window
        for i in range(0, len(token_ids)-max_len, stride):
            in_chunk= token_ids[i : i+max_len]
            tar_chunk= token_ids[i+1: i+max_len+1]
            self.input_ids.append(torch.tensor(in_chunk))
            self.target_ids.append(torch.tensor(tar_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch=4, max_len=256, stride=128, shuffle=True, drop_last=True, num_workers=0): 
    #num_workers -> how many subprocesses are used to load in //
    tokenizer= tiktoken.get_encoding("gpt2")
    dataset= GPTDatasetv1(txt, tokenizer, max_len, stride)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle= shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader