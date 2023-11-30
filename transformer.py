
import torch
import torch.nn as nn
from torch.nn import functional as F

######################## Setup

dropout = 0.0
context_length = 32

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
context_length = 32 # what is the maximum context length for predictions?
max_iters = 1
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_dimensions = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
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

######################## Helpers

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
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

######################## Transformer decoder

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        """
        Query Transformation:

        Purpose: The Query transformation learns to highlight aspects of the input that should be compared with other parts of the input. Essentially, it learns to ask the right questions.
        Learning: It adapts to focus on features that are important for drawing connections or dependencies between different elements in the input sequence. 

        For example, in a sentence, it might learn to focus on the subject or verb to understand the overall sentence structure.
        """

        self.query = nn.Linear(embedding_dimensions, head_size, bias=False)

        """
        Key Transformation:

        Purpose: The Key transformation learns to represent the input data in a way that can be effectively compared with the Queries. It's like preparing the input data to answer the questions posed by the Queries.
        Learning: This transformation learns to bring out features in the input data that are important for matching or correlating with other elements in the sequence. For instance, in language processing, it might learn to represent words in a way that highlights their syntactic roles or semantic meanings.
        """

        self.key = nn.Linear(embedding_dimensions, head_size, bias=False)

        """
        Value Transformation:

        Purpose: The Value transformation is focused on the actual content that will be used once a match or relevance is established between Queries and Keys. It represents the information that should be carried forward if a particular input element is deemed important.
        Learning: It learns to encode the input data in a way that preserves information necessary for the model's final output or subsequent processing stages. For example, it might emphasize certain aspects of a word's meaning that are crucial for understanding the sentence or for generating a response.
        """

        self.value = nn.Linear(embedding_dimensions, head_size, bias=False)

        # Register buffers and dropout for later use in forward
        
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        _batch_size, context_Length, head_size = x.shape

        """
        The first step is calculating the dot product between the Query vector for a particular word and the Key vector for every other word in the sequence. 
        This operation results in attention scores, representing the relevance or similarity between words:

        A higher dot product value indicates a higher degree of similarity or relevance. This means that the information in the word corresponding to the Key is likely important for understanding or processing the word represented by the Query.
        A lower dot product value suggests less relevance or similarity.
        """

        q = self.query(x) # batch_size, context_Length, headsize
        k = self.key(x)  # batch_size, context_Length, headsize

        attention_scores = q @ k.transpose(-2, -1) * head_size ** - 0.5 # 1) transpose to - batch_size, context_Length, context_Length 2) Normalise so softmax is more diffuse and less spiky

        """
        In a decoder block we mask out the next token with help from torch's tril function that creates a bottom right triangle of values and the rest is -inf
        """

        tril = torch.tril(torch.ones(context_Length, context_Length))
        attention_scores = attention_scores.masked_fill(tril == 0, float('-inf'))

        """
        The attention scores are normalized through a softmax function. 
        The softmax ensures that the scores for each word sum up to 1, essentially converting them into a probability distribution.
        """

        attention_scores = F.softmax(attention_scores, dim=-1)

        """
        Multiplying Scores with Value Vectors:

        Here's where the Value vectors come into play. Each word in the input sequence has a corresponding Value vector, just like they have Query and Key vectors.
        The normalized attention scores are then used to weight these Value vectors. This is done by multiplying the Value vectors by the attention scores (not by performing a dot product).
        """

        v = self.value(x)
        out = attention_scores @ v

        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dimensions, 4 * embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimensions, embedding_dimensions),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embedding_dimensions, n_head):
        # embedding_dimensions: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dimensions // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(embedding_dimensions)
        self.ln1 = nn.LayerNorm(embedding_dimensions)
        self.ln2 = nn.LayerNorm(embedding_dimensions)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerDecoderModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)
        self.position_embedding_table = nn.Embedding(context_length, embedding_dimensions)
        self.blocks = nn.Sequential(*[Block(embedding_dimensions, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dimensions) # final layer norm
        self.lm_head = nn.Linear(embedding_dimensions, vocab_size)

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
            # crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]
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
    

model = TransformerDecoderModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))