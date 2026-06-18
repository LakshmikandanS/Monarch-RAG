# Building Modern Retrieval Systems

## Introduction

Modern retrieval systems are used in search engines, recommendation systems, digital assistants, and retrieval augmented generation pipelines.

A retrieval system must balance three competing objectives:

1. Relevance
2. Speed
3. Scalability

Improving one objective often negatively impacts another.

For example, exact keyword search is extremely fast but may fail to capture semantic meaning.

This limitation motivated the development of embedding-based retrieval systems.

---

## Keyword Search

Traditional search engines relied heavily on lexical matching.

BM25 remains one of the most successful ranking algorithms ever developed.

BM25 does not understand meaning.

Instead, it measures statistical relationships between words and documents.

Advantages:

* Fast
* Interpretable
* Efficient

Disadvantages:

* Poor synonym handling
* Sensitive to wording differences

A search for "automobile" may fail to retrieve documents discussing "cars".

This weakness becomes important later when discussing semantic search.

---

## Embedding Models

Embedding models convert text into dense numerical vectors.

These vectors are designed so semantically similar concepts occupy nearby regions in vector space.

Examples:

* automobile
* car
* vehicle

Although these words are different lexically, an embedding model may place them near one another.

This capability allows retrieval systems to move beyond exact keyword matching.

Notice that the synonym problem introduced in the previous section can now be addressed.

---

## Vector Databases

A vector database stores embeddings and enables similarity search.

Popular systems include:

* LanceDB
* FAISS
* Chroma
* Qdrant

The database itself does not understand language.

Its responsibility is efficient nearest-neighbor retrieval.

The quality of retrieval therefore depends heavily on embedding quality.

This dependency becomes important when evaluating retrieval performance.

---

## Retrieval Evaluation

Retrieval systems require objective evaluation.

Several metrics are commonly used.

### Precision

Precision measures the fraction of retrieved documents that are relevant.

High precision means fewer irrelevant results.

### Recall

Recall measures the fraction of relevant documents successfully retrieved.

High recall means fewer missed documents.

### F1 Score

F1 Score combines precision and recall.

Poor chunking can reduce all three metrics simultaneously.

Notice that retrieval quality now depends not only on embeddings but also on document segmentation.

---

## Chunking Strategies

Chunking determines how information is divided before indexing.

Three common approaches include:

### Fixed Chunking

Documents are split into equally sized segments.

Advantages:

* Simple
* Fast

Disadvantages:

* May split concepts

### Sliding Window Chunking

Adjacent chunks overlap.

Advantages:

* Better context preservation

Disadvantages:

* Increased storage requirements

### Header-Aware Chunking

Document structure determines chunk boundaries.

Advantages:

* Preserves semantic organization

Disadvantages:

* Relies on well-formatted documents

The relationship between chunking quality and retrieval metrics is often underestimated.

---

## RAG Pipelines

Retrieval Augmented Generation combines retrieval systems with language models.

A typical pipeline:

1. User Query
2. Query Embedding
3. Similarity Search
4. Top-K Retrieval
5. Context Construction
6. LLM Generation

Notice that retrieval occurs before generation.

A powerful language model cannot compensate for missing context.

This observation explains why retrieval quality is frequently more important than model size.

---

## Failure Cases

Many RAG systems fail despite using strong language models.

Common causes:

* Poor chunking
* Weak embeddings
* Small retrieval depth
* Irrelevant context

Increasing model size alone rarely fixes these issues.

For example, retrieving irrelevant chunks about BM25 will not help answer questions about vector databases.

The retrieval layer must therefore be evaluated independently.

---

## Advanced Retrieval

Modern retrieval systems increasingly use hybrid search.

Hybrid search combines:

* BM25
* Vector Search

This approach benefits from both lexical matching and semantic understanding.

Notice how the synonym problem introduced earlier can still benefit from BM25 when exact terms appear.

At the same time, embeddings help retrieve semantically related content.

The strongest systems frequently combine both approaches.

---

## Summary

This document intentionally contains multiple dependency chains:

BM25 → Synonym Problem → Embeddings → Semantic Retrieval

Embeddings → Vector Databases → Similarity Search

Chunking → Precision → Recall → F1 Score

Retrieval → Top-K → Context Construction → Generation

A good retrieval system should preserve these relationships even when concepts are distributed across different sections.
