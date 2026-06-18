import numpy as np

def get_similarity(query_embedding, doc_embedding):
    return np.dot(doc_embedding, query_embedding) / (np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding))

def similarity_search(query_embedding, doc_embeddings):
    similarities = []   
    for doc_embedding in doc_embeddings:
        similarity = get_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)
    return similarities

def get_top_k_similar_documents(query_embedding, doc_embeddings, documents, k=1):
    similarities = similarity_search(query_embedding, doc_embeddings)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_documents = [documents[i] for i in top_k_indices]
    top_k={"documents": top_k_documents, "similarities": [similarities[i] for i in top_k_indices]}
    return top_k       

