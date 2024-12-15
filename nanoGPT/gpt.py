import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 20 #max_iters // 10 #500
learning_rate = 3e-4
eval_iters = 1 # 200 
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------------------------------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# use mps float32 for better performance
torch.set_default_dtype(torch.float32)
#torch.set_default_tensor_type(torch.float32)
torch.manual_seed(1337)


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # ix is the starting index of each sequence of size batch_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    # evaluate the loss on the train and validation sets
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        # head_size: dimension of queries, keys, and values
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(f"Running on {device}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import time
start_time = time.time()
for iter in range(max_iters):
    #print(f"iter {iter}/{max_iters}")
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        print("estimating loss : ", end="")
        losses = estimate_loss()
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | time {time.time()-start_time:.2f}s")
        start_time = time.time()

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("training complete")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))








# 3.222593 M parameters
# Running on mps
# estimating loss : step 0: train loss 4.2232, val loss 4.2240 | time 0.76s
# estimating loss : step 50: train loss 2.8163, val loss 2.8422 | time 6.75s
# estimating loss : step 100: train loss 2.5977, val loss 2.5992 | time 6.50s
# estimating loss : step 150: train loss 2.5246, val loss 2.5314 | time 6.46s
# estimating loss : step 200: train loss 2.4834, val loss 2.4913 | time 6.45s
# estimating loss : step 250: train loss 2.4471, val loss 2.4679 | time 6.45s
# estimating loss : step 300: train loss 2.4208, val loss 2.4402 | time 6.50s
# estimating loss : step 350: train loss 2.4023, val loss 2.4129 | time 6.50s
# estimating loss : step 400: train loss 2.3753, val loss 2.3938 | time 6.45s
# estimating loss : step 450: train loss 2.3392, val loss 2.3641 | time 6.49s
# estimating loss : step 500: train loss 2.3064, val loss 2.3387 | time 7.33s
# estimating loss : step 550: train loss 2.2541, val loss 2.2893 | time 6.45s
# estimating loss : step 600: train loss 2.2239, val loss 2.2562 | time 6.45s
# estimating loss : step 650: train loss 2.1741, val loss 2.2098 | time 6.45s
# estimating loss : step 700: train loss 2.1285, val loss 2.1683 | time 6.45s
# estimating loss : step 750: train loss 2.0623, val loss 2.1144 | time 6.54s
# estimating loss : step 800: train loss 2.0165, val loss 2.0663 | time 6.45s
# estimating loss : step 850: train loss 1.9820, val loss 2.0418 | time 6.44s
# estimating loss : step 900: train loss 1.9342, val loss 2.0017 | time 6.45s
# estimating loss : step 950: train loss 1.9064, val loss 1.9949 | time 6.56s
# estimating loss : step 1000: train loss 1.8672, val loss 1.9584 | time 6.62s
# estimating loss : step 1050: train loss 1.8428, val loss 1.9529 | time 6.52s
# estimating loss : step 1100: train loss 1.8128, val loss 1.9308 | time 6.47s
# estimating loss : step 1150: train loss 1.7813, val loss 1.9030 | time 6.46s
# estimating loss : step 1200: train loss 1.7605, val loss 1.9032 | time 6.48s
# estimating loss : step 1250: train loss 1.7526, val loss 1.8807 | time 6.47s
# estimating loss : step 1300: train loss 1.7153, val loss 1.8558 | time 6.55s
# estimating loss : step 1350: train loss 1.6983, val loss 1.8357 | time 6.47s
# estimating loss : step 1400: train loss 1.6934, val loss 1.8210 | time 6.47s
# estimating loss : step 1450: train loss 1.6774, val loss 1.8315 | time 6.48s
# estimating loss : step 1500: train loss 1.6598, val loss 1.8119 | time 6.47s
# estimating loss : step 1550: train loss 1.6579, val loss 1.8185 | time 6.48s
# estimating loss : step 1600: train loss 1.6363, val loss 1.8122 | time 6.53s
# estimating loss : step 1650: train loss 1.6293, val loss 1.7943 | time 6.46s
# estimating loss : step 1700: train loss 1.5992, val loss 1.7753 | time 6.47s
# estimating loss : step 1750: train loss 1.6057, val loss 1.7801 | time 6.47s
# estimating loss : step 1800: train loss 1.5822, val loss 1.7845 | time 6.49s
# estimating loss : step 1850: train loss 1.5950, val loss 1.7596 | time 6.48s
# estimating loss : step 1900: train loss 1.5601, val loss 1.7591 | time 6.46s
# estimating loss : step 1950: train loss 1.5698, val loss 1.7365 | time 6.46s
# estimating loss : step 2000: train loss 1.5548, val loss 1.7419 | time 6.45s
# estimating loss : step 2050: train loss 1.5367, val loss 1.7311 | time 6.45s
# estimating loss : step 2100: train loss 1.5410, val loss 1.7189 | time 6.50s
# estimating loss : step 2150: train loss 1.5325, val loss 1.7370 | time 6.46s
# estimating loss : step 2200: train loss 1.5386, val loss 1.7191 | time 6.45s
# estimating loss : step 2250: train loss 1.5216, val loss 1.7106 | time 6.45s
# estimating loss : step 2300: train loss 1.5054, val loss 1.7006 | time 6.47s
# estimating loss : step 2350: train loss 1.5150, val loss 1.6920 | time 6.56s
# estimating loss : step 2400: train loss 1.4989, val loss 1.7038 | time 6.46s
# estimating loss : step 2450: train loss 1.4949, val loss 1.6855 | time 6.45s
# estimating loss : step 2500: train loss 1.4849, val loss 1.6943 | time 6.47s
# estimating loss : step 2550: train loss 1.4906, val loss 1.6907 | time 6.46s
# estimating loss : step 2600: train loss 1.4671, val loss 1.6885 | time 6.46s
# estimating loss : step 2650: train loss 1.4726, val loss 1.6869 | time 6.47s
# estimating loss : step 2700: train loss 1.4708, val loss 1.6902 | time 6.46s
# estimating loss : step 2750: train loss 1.4514, val loss 1.6721 | time 6.46s
# estimating loss : step 2800: train loss 1.4683, val loss 1.6669 | time 6.54s
# estimating loss : step 2850: train loss 1.4496, val loss 1.6514 | time 6.49s
# estimating loss : step 2900: train loss 1.4666, val loss 1.6437 | time 6.44s
# estimating loss : step 2950: train loss 1.4525, val loss 1.6641 | time 6.44s
# estimating loss : step 3000: train loss 1.4535, val loss 1.6631 | time 6.45s
# estimating loss : step 3050: train loss 1.4222, val loss 1.6481 | time 6.46s
# estimating loss : step 3100: train loss 1.4390, val loss 1.6435 | time 6.50s
# estimating loss : step 3150: train loss 1.4425, val loss 1.6341 | time 6.44s
# estimating loss : step 3200: train loss 1.4358, val loss 1.6367 | time 6.44s
# estimating loss : step 3250: train loss 1.4310, val loss 1.6384 | time 6.45s
# estimating loss : step 3300: train loss 1.4204, val loss 1.6270 | time 6.45s
# estimating loss : step 3350: train loss 1.4232, val loss 1.6797 | time 6.47s
# estimating loss : step 3400: train loss 1.4256, val loss 1.6401 | time 6.51s
# estimating loss : step 3450: train loss 1.4014, val loss 1.6242 | time 6.45s
# estimating loss : step 3500: train loss 1.4237, val loss 1.6381 | time 6.44s
# estimating loss : step 3550: train loss 1.4177, val loss 1.6315 | time 6.44s
# estimating loss : step 3600: train loss 1.4029, val loss 1.6260 | time 6.45s
# estimating loss : step 3650: train loss 1.4023, val loss 1.6104 | time 6.45s
# estimating loss : step 3700: train loss 1.3934, val loss 1.6085 | time 6.53s
# estimating loss : step 3750: train loss 1.3936, val loss 1.6097 | time 6.45s
# estimating loss : step 3800: train loss 1.3963, val loss 1.6082 | time 6.54s
# estimating loss : step 3850: train loss 1.3759, val loss 1.6105 | time 6.46s
# estimating loss : step 3900: train loss 1.3868, val loss 1.6000 | time 6.46s
# estimating loss : step 3950: train loss 1.3926, val loss 1.6137 | time 6.46s
# estimating loss : step 4000: train loss 1.3850, val loss 1.5950 | time 6.46s
# estimating loss : step 4050: train loss 1.3701, val loss 1.5897 | time 6.46s
# estimating loss : step 4100: train loss 1.3874, val loss 1.5862 | time 6.46s
# estimating loss : step 4150: train loss 1.3744, val loss 1.5937 | time 6.47s
# estimating loss : step 4200: train loss 1.3761, val loss 1.6066 | time 6.48s
# estimating loss : step 4250: train loss 1.3676, val loss 1.6072 | time 6.53s
# estimating loss : step 4300: train loss 1.3664, val loss 1.5901 | time 6.47s
# estimating loss : step 4350: train loss 1.3654, val loss 1.6045 | time 6.45s
# estimating loss : step 4400: train loss 1.3571, val loss 1.5738 | time 6.46s
# estimating loss : step 4450: train loss 1.3671, val loss 1.6063 | time 6.46s
# estimating loss : step 4500: train loss 1.3642, val loss 1.5843 | time 6.47s
# estimating loss : step 4550: train loss 1.3466, val loss 1.5924 | time 6.52s
# estimating loss : step 4600: train loss 1.3471, val loss 1.5715 | time 6.50s
# estimating loss : step 4650: train loss 1.3431, val loss 1.5774 | time 6.49s
# estimating loss : step 4700: train loss 1.3632, val loss 1.5871 | time 6.46s
# estimating loss : step 4750: train loss 1.3447, val loss 1.5750 | time 6.48s
# estimating loss : step 4800: train loss 1.3595, val loss 1.5830 | time 6.52s
# estimating loss : step 4850: train loss 1.3448, val loss 1.5477 | time 6.51s
# estimating loss : step 4900: train loss 1.3516, val loss 1.5556 | time 6.52s
# estimating loss : step 4950: train loss 1.3334, val loss 1.5640 | time 6.47s
# estimating loss : step 4999: train loss 1.3376, val loss 1.5832 | time 6.37s
# training complete


# WARWICKBHORY:
# A prite thousant?

# Putcharged here, man:
# Certay, he wry mady, so, in't now that give your will life,
# I for reveigng me band.

# HORTENSIO:
# O, was my mother, my less and humb igpster?

# MERCATINGS:
# Dead liber, though Harvants, best! as the true.

# WARWICKING EDWARD IV:
# Who say is Isabell'd my grace:
# Kind we not Wilw you have your life
# The patter foone toes me malicator.

# HERMIONE:
# Ay, so discrecord governt angerious brach selvess.

# BARNAMy LANUS:
# Doth Duke of orgy; the Lord BenHarRD:
# H