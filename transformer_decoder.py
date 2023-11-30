
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, context_length, embedding_dimensions, dropout):
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

    def __init__(self, num_heads, head_size, context_length, embedding_dimensions, dropout):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size, context_length, embedding_dimensions, dropout) for _ in range(num_heads)])
        self.linear_layer_projection = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        Multiple heads are concatinated and fed through a linear layer, the linear layer will learn which head it should use with respect to the input.
        Different heads can learn different nuances of the input.
        """

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear_layer_projection(out))

        return out
    
class FeedFoward(nn.Module):

    def __init__(self, embedding_dimensions, dropout):
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

    def __init__(self, embedding_dimensions, n_head, context_length, dropout):

        # embedding_dimensions: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dimensions // n_head
        self.multi_head_attention = MultiHeadAttention(n_head, head_size, context_length, embedding_dimensions, dropout)
        self.feed_forward = FeedFoward(embedding_dimensions, dropout)
        self.layer_normalisation_1 = nn.LayerNorm(embedding_dimensions)
        self.layer_normalisation_2 = nn.LayerNorm(embedding_dimensions)

    def forward(self, x):

        x = x + self.multi_head_attention(self.layer_normalisation_1(x))
        x = x + self.feed_forward(self.layer_normalisation_2(x))

        return x

class TransformerDecoderModel(nn.Module):

    def __init__(self, vocab_size, context_length, embedding_dimensions, n_head, n_layer, dropout, device):
        super().__init__()

        self.context_length = context_length
        self.device = device

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)
        self.position_embedding_table = nn.Embedding(self.context_length, embedding_dimensions)
        self.transformer_blocks = nn.Sequential(*[Block(embedding_dimensions, n_head=n_head, context_length=context_length, dropout=dropout) for _ in range(n_layer)])
        self.final_layer_normalisation = nn.LayerNorm(embedding_dimensions) # final layer norm
        self.final_linear_head = nn.Linear(embedding_dimensions, vocab_size)

    def forward(self, idx, targets=None):
        B, vocab_size = idx.shape

        # We input token_embedding + positional_embedding to the transformer
        token_embedding = self.token_embedding_table(idx) # (B,T,C)
        positional_embedding = self.position_embedding_table(torch.arange(vocab_size, device=self.device)) # (T,C)
        x = token_embedding + positional_embedding # (B,T,C)

        # Go through stacked transformer blocks
        x = self.transformer_blocks(x) # (B,T,C)

        # Normalise the final layer
        x = self.final_layer_normalisation(x) # (B,T,C)

        # Get logits out
        logits = self.final_linear_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, vocab_size, C = logits.shape
            logits = logits.view(B*vocab_size, C)
            targets = targets.view(B*vocab_size)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context_length tokens
            idx_cond = idx[:, -self.context_length:]
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
    
