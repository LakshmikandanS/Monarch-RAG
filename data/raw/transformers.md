# Transformer Architectures and Attention

## Introduction

Before Transformers, sequence-to-sequence tasks relied heavily on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. 

While these older models were effective, they suffered from the bottleneck problem. They had to process information sequentially, word by word.

The bottleneck problem made it difficult for models to remember information from the beginning of a long paragraph by the time they reached the end. 

This limitation was entirely resolved by the introduction of the self-attention mechanism.

---

## The Attention Mechanism

Self-attention allows a model to look at every single word in a sequence simultaneously and determine which words are most relevant to each other, regardless of distance.

Three key vectors are created for every token:
* Query (Q)
* Key (K)
* Value (V)

The relationship between these vectors is calculated using scaled dot-product attention. 

### Mathematical Formulation

The attention score is computed by taking the dot product of the query and key, scaling it, applying a softmax function, and multiplying by the value:

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Notice that this mathematical operation relies heavily on the matrix multiplications discussed previously in the context of hardware accelerators.

---

## Multi-Head Attention

A single attention mechanism might focus too heavily on one specific relationship between words.

Multi-head attention solves this by running multiple attention mechanisms in parallel. 

1. The input vectors are projected into multiple lower-dimensional spaces.
2. Attention is computed independently for each "head".
3. The results are concatenated.
4. A final linear transformation is applied.

This allows the model to simultaneously attend to different aspects of the text, such as grammar, vocabulary, and context.

---

## Implementation Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Multiply by values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```