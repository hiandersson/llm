import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f'text length {len(text)}')

characters = sorted(list(set(text)))
vocab_size = len(characters)
#print(''.join(characters))
#print(vocab_size)

string_to_integer = { ch:i for i,ch in enumerate(characters) }
integer_to_string = { i:ch for i,ch in enumerate(characters) }
encode = lambda s: [string_to_integer[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([integer_to_string[i] for i in l]) # decoder: take a list of integers, output a string

#print(encode("hii there"))
#print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:100])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

context_length = 8
batch_size = 4

def get_random_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x, y


xb, yb = get_random_batch('train')
"""
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
"""

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x_idx, y_targets=None):

        print('x_idx {}'.format(x_idx))
        print('y_targets {}'.format(y_targets))

        # x_idx and y_targets are both (Batch, Time) tensor of integers
        logits = self.token_embedding_table(x_idx) # B,T,C = (Batch = batch_size, Time = context_length, Channel = vocab_size)

        print('logits {}'.format(logits))
        
        if y_targets is None:
            loss = None
        else:

            # Pytorch wants cross_entropy in a dfferent format than just B,T,C --> B*T, C
            Batch_Size, Context_Length, Vocab_Size = logits.shape
            logits = logits.view(Batch_Size*Context_Length, Vocab_Size)
            y_targets = y_targets.view(Batch_Size*Context_Length)

            # Loss
            loss = F.cross_entropy(logits, y_targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
