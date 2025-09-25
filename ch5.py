import torch
import tiktoken
import matplotlib.pyplot as plt
from basic import create_dataloader_v1
from GPT_architecture import GPTModel
from GPT_architecture import Generate_text

GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256, # Shortened from 1024 to make it easier to train on a laptop
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
}

def text_to_token_ids(txt, tokenizer):
    encoded= tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
    encoded_tensor= torch.tensor(encoded).unsqueeze(0) # add dimension at position 0
    return encoded_tensor

def token_ids_to_txt(token_ids, tokenizer):
    flat= token_ids.squeeze(0) # Remove batch dimension
    return tokenizer.decode(flat.tolist())

def cross_entropy(input_batch, target_batch, model, device, num_batches=None):
    input_batch= input_batch.to(device)
    target_batch= target_batch.to(device)
    logits= model(input_batch)
    loss= torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches= min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss= cross_entropy(input_batch, target_batch, model, device)
            total_loss+=loss.item()
        else:
            break
    return total_loss/num_batches

def evaulate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # to stop dropout
    # no grad to stop gradient calculation 
    with torch.no_grad(): 
        train_loss= calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss= calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size= model.pos_emb.weight.shape[0]
    encoded= text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids= Generate_text(model= model, idx= encoded, max_new_tokens=50, context_size=context_size)
        decoded_text= token_ids_to_txt(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_fr, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen= [], [], []
    tokens_seen, global_step= 0,-1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
         optimizer.zero_grad()
         loss = cross_entropy(input_batch, target_batch, model, device)
         loss.backward()
         optimizer.step()
         tokens_seen+=input_batch.numel()
         global_step+=1

         if global_step % eval_fr ==0:
            train_loss, val_loss = evaulate_model(model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep{epoch+1}(Step {global_step:06d}):"
                  f"Train loss{train_loss:.3f}"
                  f"Val loss{val_loss:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond= idx[:, -context_size:]
        with torch.no_grad():
            logits= model(idx_cond)
        logits= logits[:,-1,:]
        if top_k is not None:
            top_logits, _=torch.topk(logits, top_k)
            min_val= top_logits[:,-1]
            logits= torch.where(
                logits<min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits= logits/temperature
            probs= torch.softmax(logits, dim=-1)
            idx_next= torch.multinomial(probs, num_samples=1)
        else:
            idx_next= torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx= torch.cat((idx, idx_next), dim=1)
    return idx

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()