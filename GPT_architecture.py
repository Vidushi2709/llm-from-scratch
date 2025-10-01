import torch
import torch.nn as nn
import tiktoken
from attention import MultiHeadAttention
from torch.nn.utils.rnn import pad_sequence

GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
}

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x
    
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg)
                  for _ in range(cfg["n_layers"])]
                )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias = False
                )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
                )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False matches nn.LayerNorm
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
    
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
                # Implement 5 layers

                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
                ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# A function to compute gradients
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absoute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
         d_in=cfg["emb_dim"],
         d_out=cfg["emb_dim"],          
         context_length=cfg["context_length"],
         dropout=cfg["drop_rate"],
         num_heads=cfg["n_heads"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Multi-head attention + residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward + residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class TransformerBlockSeperateDropout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn= MultiHeadAttention(
                    d_in=cfg["emb_dim"],
                    d_out=cfg["emb_dim"],
                    context_length=cfg["context_length"],
                    num_heads=cfg["n_heads"],
                    dropout=cfg["attn_drop_rate"],
                    qkv_bias=cfg["qkv_bias"]
        )
        self.ff= FeedForward(cfg)
        self.norm1= LayerNorm(cfg['emb_dim'])
        self.norm2= LayerNorm(cfg['emb_dim'])
        self.drop_shortcut= nn.Dropout(cfg["shortcut_drop_rate"])
     
    def forward(self,x):
         shortcut= x
         x= self.norm1(x)
         x = self.attn(x)
         x = x+ shortcut

         shortcut=x
         x= self.norm2(x)
         x= self.attn(x)
         x=x+shortcut
         return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb= nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb= nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb= nn.Dropout(cfg['drop_rate'])
        self.trf_blocks= nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm= LayerNorm(cfg['emb_dim'])
        self.out_head= nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias= False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds= self.tok_emb(in_idx)

        pos_embeds= self.pos_emb(
            torch.arange(seq_len, device= in_idx.device)
        )

        x= tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits= self.out_head(x)

        return logits

def createModelAndCalculateSize(conf):
    model= GPTModel(conf)
    total_params= sum(p.numel() for p in model.parameters())
    total_size_in_bytes= total_params*4
    total_size_in_mbs= total_size_in_bytes/(1024*1024)
    return total_params, total_size_in_mbs

def Generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond= idx[:,-context_size:]
        with torch.no_grad():
            logits= model(idx_cond)
        logits= logits[:,-1,:]
        probas= torch.softmax(logits, dim=-1)
        idx_next= torch.argmax(probas, dim=-1, keepdim=True)
        idx= torch.cat((idx, idx_next), dim=1)
    
    return idx