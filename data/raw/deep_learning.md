# Neural Networks in Modern Deep Learning

## Introduction

Deep learning systems are built from layers of interconnected mathematical operations. These operations allow models to learn representations directly from data instead of relying on manually engineered features.

A neural network consists of neurons organized into layers. The most common architecture is the feedforward neural network, where information moves from input to output without cycles.

The success of modern deep learning is largely attributed to three factors:

1. Large datasets
2. Powerful hardware
3. Improved optimization algorithms

The third factor becomes particularly important when discussing stochastic gradient descent.

---

## Gradient Descent

Gradient descent is an optimization algorithm used to minimize a loss function.

The basic update rule modifies model parameters in the opposite direction of the gradient.

Key components:

* Learning rate
* Loss function
* Gradient
* Parameters

The learning rate determines how large each update step should be.

If the learning rate is too large, optimization may diverge.

If the learning rate is too small, convergence may become extremely slow.

---

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) improves computational efficiency by using a subset of the training data during each optimization step.

Rather than computing gradients using the entire dataset, SGD estimates gradients from smaller samples.

Advantages:

* Faster iterations
* Lower memory consumption
* Better scalability

Disadvantages:

* Noisy gradients
* Less stable convergence

Notice that SGD still follows the same optimization principles discussed in the previous section on Gradient Descent.

---

## Mini-Batch Gradient Descent

Mini-batch gradient descent is a compromise between full-batch gradient descent and stochastic gradient descent.

A mini-batch may contain:

* 16 samples
* 32 samples
* 64 samples
* 128 samples

The choice depends on hardware constraints and model architecture.

Modern GPU training commonly uses mini-batches because they efficiently utilize parallel computation resources.

The term "parallel computation resources" will become important when discussing modern accelerators.

---

## Hardware Acceleration

Training deep neural networks requires massive numbers of matrix multiplications.

Hardware accelerators include:

### CPU

Central Processing Units are flexible but generally slower for large-scale tensor operations.

### GPU

Graphics Processing Units provide thousands of parallel execution units.

These parallel execution units make matrix multiplication significantly faster.

### NPU

Neural Processing Units are specialized processors designed specifically for AI workloads.

Unlike general-purpose CPUs, NPUs dedicate silicon to neural network operations.

This specialization often improves power efficiency.

Notice that "parallel execution units" were introduced in the previous section before discussing GPUs directly.

---

## Embeddings

Embeddings transform discrete objects into dense numerical vectors.

Words with similar meanings tend to occupy nearby regions in vector space.

Examples:

* king
* queen
* prince
* princess

Embeddings are foundational for retrieval systems because similarity can be measured mathematically.

Cosine similarity is commonly used.

The concept of vector space becomes important again in the retrieval section.

---

## Retrieval Systems

Retrieval systems attempt to locate relevant information from large collections of documents.

A typical retrieval pipeline:

1. Query
2. Embedding
3. Similarity Search
4. Ranking
5. Context Construction

The phrase "vector space" mentioned earlier is directly related to embedding-based retrieval.

Without embeddings, semantic retrieval becomes difficult.

---

## Code Example

```python
def cosine_similarity(a, b):
    numerator = sum(x*y for x, y in zip(a, b))

    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5

    return numerator / (norm_a * norm_b)
```

This implementation computes cosine similarity between two vectors.

The mathematical intuition behind this code depends on understanding embeddings from the previous section.

---

## Retrieval Metrics

Several metrics are used to evaluate retrieval quality.

### Precision

Precision measures how many retrieved documents are relevant.

### Recall

Recall measures how many relevant documents were successfully retrieved.

### F1 Score

F1 Score combines precision and recall into a single metric.

Poor chunking strategies often reduce retrieval quality even when embedding models are strong.

---

## Summary

This document intentionally contains:

* Long-form explanations
* Cross-references
* Dependency chains
* Code blocks
* Nested headers

The goal is to test whether a chunking strategy preserves relationships between concepts such as:

Gradient Descent → SGD → Mini-Batch SGD

and

Embeddings → Vector Space → Retrieval → Cosine Similarity

A good chunking strategy should maintain these relationships during retrieval.
