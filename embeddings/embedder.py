from fastembed import TextEmbedding

def load_embedding_model():
    return TextEmbedding()

def embed_documents(documents, model):
    return list(
        model.embed(
            [doc["content"] for doc in documents]
        )
    )

def embed_query(query, model):
    return list(model.embed([query]))[0]