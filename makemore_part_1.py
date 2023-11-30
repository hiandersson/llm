import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
print(words[:10])

##### Count characters in a bigram model

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
string_to_integer = {s:i+1 for i,s in enumerate(chars)}
string_to_integer['.'] = 0
integer_to_string = {i:s for s,i in string_to_integer.items()}

# Sample for every character, what character comes next

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = string_to_integer[ch1]
		ix2 = string_to_integer[ch2]
		N[ix1, ix2] += 1
		
# Show counts

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
	for j in range(27):
		chstr = integer_to_string[i] + integer_to_string[j]
		plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
		plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')

##### Sample names from a probability disitribution

# Create probabilies

P = (N+1).float() # Smooth probabilities by adding 1
P /= P.sum(1, keepdims=True) # Normalise - the sum of probabilities becomes 1

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
	out = []
	character_index = 0 # start with .

	while True:

		p = P[character_index] # get the row for the character, with probabilities

		character_index = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # sample from probability distribution

		out.append(integer_to_string[character_index]) # get and append

		if character_index == 0: # break if reaching .
			break

	print(''.join(out))

##### Train a neural net to learn the probabilities of which character comes after next

xs, ys = [], []

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]): 
		character_index_x = string_to_integer[ch1]
		character_index_y = string_to_integer[ch2]
		xs.append(character_index_x)
		ys.append(character_index_y)
	
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of elements {}'.format(num))

neurons = 27

Generator = torch.Generator().manual_seed(1)
Weights = torch.randn((27, neurons), generator=Generator, requires_grad=True)

learning_rate = 50

# Maximize the probability of Y given the parameters of the model
for k in range(1):

	### Forward

	## Inputs and logits

	# One hot encode X
	one_one_encoded_x = F.one_hot(xs, num_classes=27).float()

	# In one batch we can calculate all inputs x all weights. 
	# We interpret this as the predicted logarithm of counts.
	logits = one_one_encoded_x @ Weights

	## Softmax

	# Logits are the output from the linear layer.
	# We dont want to work with negative values when calculating probabilities later, so lets exponentiate them. Negative number become 0 - 1 and positive number become 1 and above. 
	# We interpret these as fake "counts" in regards to the bigram model counts above.
	counts = logits.exp()

	# Go from counts to normalised probabilities
	probabilities = counts / counts.sum(1, keepdims=True)

	## Calculate loss. 
	
	# Loss / error is calculated by getting the probabilities of character b coming after a by 
	# Loss should be minimized so we is - (otherwise its negative).
	loss = -probabilities[torch.arange(num), ys].log().mean()
	print(loss.item())

	### Backward

	## Reset and get gradients

	Weights.grad = None
	loss.backward()

	## Apply gradients with some learning rate

	Weights.data += -learning_rate * Weights.grad


