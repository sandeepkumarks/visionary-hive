# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from preprocessor import Preprocessor
# from retrieval_model import RetrievalModel
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import logging
#
# logging.basicConfig(level=logging.INFO)
#
# # Preprocess the unstructured policy documents
# preprocessor = Preprocessor(input_dir="documents", output_dir="documents_structured")
# preprocessor.preprocess_documents()  # Preprocess the documents before loading them into the retrieval model
#
# # Initialize the retrieval model
# retrieval_model = RetrievalModel(structured_dir="documents_structured")
#
# # Initialize the generative model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# generative_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
#
# # Move the generative model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generative_model.to(device)
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Synonym handling for queries
# def preprocess_query(user_query):
#     synonyms = {
#         "compensation": ["salary", "pay", "wages"],
#         "health": ["well-being", "medical", "insurance"],
#         # Add more synonyms as necessary
#     }
#
#     # Replace query terms with potential synonyms
#     for word, synonym_list in synonyms.items():
#         for synonym in synonym_list:
#             if synonym in user_query:
#                 user_query = user_query.replace(synonym, word)
#
#     return user_query
#
# class Query(BaseModel):
#     user_query: str
#
# @app.post("/answer")
# async def answer_query(query: Query):
#     user_query = preprocess_query(query.user_query.strip())
#
#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
#
#     # Retrieve top 3 relevant contexts from the policy document
#     retrieved_info = retrieval_model.retrieve(user_query, top_k=5)
#
#     if not retrieved_info:
#         return {"answer": "I'm not sure about that policy."}
#
#     # Combine retrieved sections for better context
#     context = " ".join(retrieved_info)
#
#     # Create a prompt with context for the generative model
#     prompt = (
#         "You are an assistant that provides detailed answers based on company policies.\n\n"
#         f"Question: {user_query}\n"
#         f"Policy Context: {context}\n"
#         "Answer in detail:"
#     )
#
#     # Tokenize and generate the answer
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#     output = generative_model.generate(
#         inputs.input_ids,
#         max_new_tokens=100,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True
#     )
#     answer = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     return {"answer": answer}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from preprocessor import Preprocessor
from retrieval_model import RetrievalModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

logging.basicConfig(level=logging.INFO)

# Preprocess the unstructured policy documents
preprocessor = Preprocessor(input_dir="documents", output_dir="documents_structured")
preprocessor.preprocess_documents()  # Preprocess the documents before loading them into the retrieval model

# Initialize the retrieval model
retrieval_model = RetrievalModel(structured_dir="documents_structured")

# Initialize the generative model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generative_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Move the generative model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generative_model.to(device)

# Initialize FastAPI app
app = FastAPI()

# Synonym handling for queries
def preprocess_query(user_query):
    synonyms = {
        "compensation": ["salary", "pay", "wages"],
        "health": ["well-being", "medical", "insurance"],
        # Add more synonyms as necessary
    }

    # Replace query terms with potential synonyms
    for word, synonym_list in synonyms.items():
        for synonym in synonym_list:
            if synonym in user_query:
                user_query = user_query.replace(synonym, word)

    return user_query

class QueryModel(BaseModel):
    user_query: str

@app.post("/answer")
async def answer_query(query: QueryModel, isRAGEnabled: bool = Query(True)):
    user_query = preprocess_query(query.user_query.strip())

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if isRAGEnabled:
        # RAG Enabled: Use both retrieval model and generative model
        # Retrieve top 3 relevant contexts from the policy document
        retrieved_info = retrieval_model.retrieve(user_query, top_k=5)

        if not retrieved_info:
            return {"answer": "I'm not sure about that policy."}

        # Combine retrieved sections for better context
        context = " ".join(retrieved_info)

        # Create a prompt with context for the generative model
        prompt = (
            "You are an assistant that provides detailed answers based on company policies.\n\n"
            f"Question: {user_query}\n"
            f"Policy Context: {context}\n"
            "Answer in detail:"
        )
    else:
        # RAG Disabled: Use only the generative model without retrieval
        prompt = (
            # "You are an assistant that provides answers to any question.\n\n"
            "You are an assistant that provides detailed answers based on company policies.\n\n"
            f"Question: {user_query}\n"
            "Answer:"
        )

    # Tokenize and generate the answer
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = generative_model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
