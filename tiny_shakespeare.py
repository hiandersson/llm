
import torch
from transformer_decoder import TransformerDecoderModel

class TinyShakespeare:

	def __init__(self, learning_rate, max_iters, eval_interval, eval_iters, context_length, batch_size, embedding_dimensions, n_head, n_layer, dropout) -> None:

		torch.manual_seed(1337)

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.learning_rate = learning_rate
		self.max_iters = max_iters
		self.eval_interval = eval_interval
		self.eval_iters = eval_iters
		self.context_length = context_length
		self.batch_size = batch_size
		self.embedding_dimensions = embedding_dimensions
		self.n_head = n_head
		self.n_layer = n_layer
		self.dropout = dropout

	def create_model(self):

		self.model = TransformerDecoderModel(
			vocab_size = self.vocab_size, 
			context_length = self.context_length,
			embedding_dimensions = self.embedding_dimensions,
			n_head = self.n_head,
			n_layer = self.n_layer,
			dropout = self.dropout,
			device = self.device)
		
		self.m = self.model.to(self.device)

		print(sum(p.numel() for p in self.m.parameters())/1e6, 'Model parameters')

	def load(self):

		with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
			self.text = f.read()

		chars = sorted(list(set(self.text)))
		self.vocab_size = len(chars)

		stoi = { ch:i for i,ch in enumerate(chars) }
		itos = { i:ch for i,ch in enumerate(chars) }
		self.encode = lambda s: [stoi[c] for c in s]
		self.decode = lambda l: ''.join([itos[i] for i in l]) 

		data = torch.tensor(self.encode(self.text), dtype=torch.long)
		n = int(0.9*len(data)) # first 90% will be train, rest val
		self.train_data = data[:n]
		self.val_data = data[n:]

	def get_batch(self, split):

		# generate a small batch of data of inputs x and targets y
		data = self.train_data if split == 'train' else self.val_data
		ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
		x = torch.stack([data[i:i + self.context_length] for i in ix])
		y = torch.stack([data[i+1:i + self.context_length+1] for i in ix])
		x, y = x.to(self.device), y.to(self.device)

		return x, y

	@torch.no_grad()
	def estimate_loss(self):
		out = {}
		self.model.eval()
		for split in ['train', 'val']:
			losses = torch.zeros(self.eval_iters)
			for k in range(self.eval_iters):
				X, Y = self.get_batch(split)
				logits, loss =self. model(X, Y)
				losses[k] = loss.item()
			out[split] = losses.mean()
		self.model.train()
		return out

	def train(self):

		# create a PyTorch optimizer
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

		for iter in range(self.max_iters):

			# every once in a while evaluate the loss on train and val sets
			if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
				losses = self.estimate_loss()
				print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

			# sample a batch of data
			xb, yb = self.get_batch('train')

			# evaluate the loss
			_logits, loss = self.model(xb, yb)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

	def generate(self):
		context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
		return self.decode(self.m.generate(context, max_new_tokens=2000)[0].tolist())
	
# Experiments

tiny = TinyShakespeare(
	learning_rate = 1e-3, 
	max_iters = 1, 
	eval_interval = 100,
	eval_iters = 200,
	context_length = 32,
	batch_size = 1,
	embedding_dimensions=64,
	n_head = 4, 
	n_layer = 4, 
	dropout = 0.2)

tiny.load()

tiny.create_model()

tiny.train()

print(tiny.generate())