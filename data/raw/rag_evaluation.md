# Advanced RAG Evaluation Metrics

## The Limits of Statistical Retrieval

Traditional retrieval metrics like Precision, Recall, and F1 Score measure whether the correct document was fetched from the database.

However, in a Retrieval Augmented Generation pipeline, retrieving the right document is only half the battle.

If the language model generates an incorrect answer despite having the correct document, the system has failed. 

This necessitates the use of generative evaluation metrics.

---

## Types of Generation Failures

When an LLM generates a bad response, it usually falls into one of two categories:

### Factual Hallucination
The model states something that is objectively false in the real world. 

### Faithfulness Hallucination
The model states something that might be true in the real world, but is NOT supported by the retrieved context.

In a RAG system, faithfulness hallucinations are far more dangerous because the model is ignoring the retrieval database and relying on its parametric memory instead.

---

## The RAG Triad

Modern RAG evaluation relies on three core generative metrics. These are fundamentally different from BM25 or semantic similarity scores.

### Context Relevance
Measures whether the retrieved chunks actually contain information useful for answering the user's query.

If context relevance is low, it indicates a failure in the embedding model or the chunking strategy.

### Groundedness (Faithfulness)
Measures whether the final LLM response can be directly traced back to the retrieved context. 

High groundedness means the model did not invent outside information.

### Answer Relevance
Measures whether the final LLM response directly addresses the user's original prompt.

A response can be perfectly grounded in the context, but still fail to answer what the user actually asked.

---

## LLM-as-a-Judge

Calculating generative metrics programmatically is difficult because language is highly variable. 

Instead, developers use "LLM-as-a-Judge".

In this paradigm, a strong language model (like GPT-4 or Gemini) is given the prompt, the retrieved context, and the generated answer. 

It is then asked to grade the system based on the RAG Triad.

Advantages:
* Correlates highly with human judgment
* Scales automatically across thousands of test cases
* Does not require manual data labeling

Disadvantages:
* Expensive to run at scale
* Susceptible to position bias (the judge favors answers that put important information at the very beginning)

Notice that position bias here is a completely different concept than the positional encoding used in Transformer architectures.