import os
import ollama

from query_rewriting.prompts import RAG_prompt

os.environ["NO_PROXY"] = "http://localhost:11434"

client = ollama.Client(host="http://127.0.0.1:11434")

def rewrite_query(query, model_name="phi3:latest"):
    response = client.generate(
        model=model_name,
        prompt=RAG_prompt.format(query=query),
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 100,
        },
    )

    # print("Thinking:", response.thinking[:200] if response.thinking else "None")
    # print("Response:", response.response)
    # print("Done Reason:", response.done_reason)
    # print("Eval Count:", response.eval_count)
    
    return response.response.strip()