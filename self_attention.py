import torch
import torch.nn as nn
from torch.nn import functional as F

# version 4: self-attention!

torch.manual_seed(1337)
batch_Size, context_Length, vocab_Size = 1, 8, 32 # batch size (4), time (context length 8), channels (vocab size 32)

x = torch.randn(batch_Size, context_Length, vocab_Size)

head_size = 16

"""
Query Transformation:

Purpose: The Query transformation learns to highlight aspects of the input that should be compared with other parts of the input. Essentially, it learns to ask the right questions.
Learning: It adapts to focus on features that are important for drawing connections or dependencies between different elements in the input sequence. 

For example, in a sentence, it might learn to focus on the subject or verb to understand the overall sentence structure.
"""

query = nn.Linear(vocab_Size, head_size)

"""
Key Transformation:

Purpose: The Key transformation learns to represent the input data in a way that can be effectively compared with the Queries. It's like preparing the input data to answer the questions posed by the Queries.
Learning: This transformation learns to bring out features in the input data that are important for matching or correlating with other elements in the sequence. For instance, in language processing, it might learn to represent words in a way that highlights their syntactic roles or semantic meanings.
"""

key = nn.Linear(vocab_Size, head_size)

"""
Value Transformation:

Purpose: The Value transformation is focused on the actual content that will be used once a match or relevance is established between Queries and Keys. It represents the information that should be carried forward if a particular input element is deemed important.
Learning: It learns to encode the input data in a way that preserves information necessary for the model's final output or subsequent processing stages. For example, it might emphasize certain aspects of a word's meaning that are crucial for understanding the sentence or for generating a response.
"""

value = nn.Linear(vocab_Size, head_size)

"""
The first step is calculating the dot product between the Query vector for a particular word and the Key vector for every other word in the sequence. 
This operation results in attention scores, representing the relevance or similarity between words:

A higher dot product value indicates a higher degree of similarity or relevance. This means that the information in the word corresponding to the Key is likely important for understanding or processing the word represented by the Query.
A lower dot product value suggests less relevance or similarity.
"""

q = query(x) # batch_Size, context_Length, headsize
k = key(x)  # batch_Size, context_Length, headsize

weights = q @ k.transpose(-2, -1) * head_size ** - 0.5 # 1) transpose to - batch_Size, context_Length, context_Length 2) Normalise so softmax is more diffuse and less spiky

"""
In a decoder block we mask out the next token with help from torch's tril function that creates a bottom right triangle of values and the rest is -inf
"""

tril = torch.tril(torch.ones(context_Length, context_Length))
weights = weights.masked_fill(tril == 0, float('-inf'))

"""
The attention scores are normalized through a softmax function. 
The softmax ensures that the scores for each word sum up to 1, essentially converting them into a probability distribution.
"""

weights = F.softmax(weights, dim=-1)

"""
Multiplying Scores with Value Vectors:

Here's where the Value vectors come into play. Each word in the input sequence has a corresponding Value vector, just like they have Query and Key vectors.
The normalized attention scores are then used to weight these Value vectors. This is done by multiplying the Value vectors by the attention scores (not by performing a dot product).
"""

v = value(x)
out = weights @ v

print(weights)