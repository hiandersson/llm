import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import random

# read in all the words
words = open('names.txt', 'r').read().splitlines()
print(words[:8])

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
	X, Y = [], []
	for w in words:

		context = [0] * block_size
		for ch in w + '.':
			ix = stoi[ch]
			X.append(context)
			Y.append(ix)
			#print(''.join(itos[i] for i in context), '--->', itos[ix])
			context = context[1:] + [ix] # crop and append

	X = torch.tensor(X)
	Y = torch.tensor(Y)
	print(X.shape, Y.shape)
	return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtrain, Ytrain = build_dataset(words[:n1])
Xvalidation, Yvalidation = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

### Neural network

embedding_dimensions = 10
embedding_in_size = block_size * embedding_dimensions
hidden_layer_size = 200
vocab_size = 27

g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, embedding_dimensions), generator=g)

W1 = torch.randn((embedding_in_size, hidden_layer_size), generator=g)
b1 = torch.randn(hidden_layer_size, generator=g)

W2 = torch.randn((hidden_layer_size, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]

### Train

batch_size = 32

for p in parameters:
	p.requires_grad = True

learning_rate = 0.1

for i in range(10000):
  
	## Get batch
	#  Sample random integers batch_size times, to get a training batch
	ix = torch.randint(0, Xtrain.shape[0], (batch_size,))
	
	## Forward pass

	# Embedding layer
	emb = C[Xtrain[ix]]

	# Hidden layer - Concatenate 
	hidden_layer = torch.tanh(emb.view(-1, embedding_in_size) @ W1 + b1) 

	# Logits layer
	logits = hidden_layer @ W2 + b2 

	# Loss
	loss = F.cross_entropy(logits, Ytrain[ix])
	
	## Backward pass

	# Get gradients
	for p in parameters:
		p.grad = None
	loss.backward()
	
	# Update parameters
	for p in parameters:
		p.data += -learning_rate * p.grad

# Final loss on the whole set
emb = C[Xtrain] 
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) 
logits = h @ W2 + b2 
fina_loss = F.cross_entropy(logits, Ytrain)
print(fina_loss)
	