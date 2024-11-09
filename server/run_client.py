from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
retrieval_model = RetrievalModel(embedding_model_name = "all-mpnet-base-v2", structured_dir="documents_structured")

# Initialize the generative model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
generative_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Move the generative model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generative_model.to(device)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Update this if your frontend URL changes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Synonym handling for queries
def preprocess_query(user_query):
    synonyms = {
        # "election": ["vote", "poll", "ballot"],
        # "victory": ["win", "success", "triumph"],
        # "state": ["region", "territory", "constituency"],
        # "battleground state": ["swing state", "key state", "crucial state"],
        # "candidate": ["contender", "nominee", "politician"],
        # "votes": ["ballots", "tallies", "counts"],
        # "campaign": ["election drive", "political campaign", "run for office"],
        # "margin": ["lead", "advantage", "difference"],
        # "president": ["head of state", "leader", "commander-in-chief"],
        # "re-election": ["second term", "return to office", "another term"]

        # Election-related synonyms
        "election": ["vote", "poll", "ballot", "elections", "election process"],
        "victory": ["win", "success", "triumph", "landslide", "conquest"],
        "state": ["region", "territory", "constituency", "province", "district"],
        "battleground state": ["swing state", "key state", "crucial state", "competitive state"],
        "candidate": ["contender", "nominee", "politician", "aspirant"],
        "votes": ["ballots", "tallies", "counts", "votings"],
        "campaign": ["election drive", "political campaign", "run for office", "election effort"],
        "margin": ["lead", "advantage", "difference", "gap", "edge"],
        "president": ["head of state", "leader", "commander-in-chief", "chief executive"],
        "re-election": ["second term", "return to office", "another term", "renewal"],
        "electoral votes": ["electoral tally", "elector count", "college votes"],

        # Cricket-related synonyms
        "match": ["game", "fixture", "encounter", "contest"],
        "century": ["hundred", "ton", "three-figure score", "100 runs"],
        "runs": ["scores", "points", "tallies"],
        "wickets": ["dismissals", "scalps", "outs"],
        "bowler": ["pacer", "spinner", "seamer", "cricketer"],
        "batsman": ["batter", "striker", "hitter", "player"],
        "overs": ["rounds", "innings sections", "bowling sequences"],
        "innings": ["batting session", "game segment", "innings phase"],
        "team": ["side", "squad", "unit", "line-up"],
        "series": ["tournament", "competition", "fixture sequence"],
        "spinner": ["slow bowler", "turner", "spin specialist"],
        "pacers": ["fast bowlers", "speedsters", "quicks"],
        "score": ["tally", "result", "total"],
        "opener": ["starting batsman", "top-order batter", "first batter"],
        "captain": ["skipper", "leader", "team head"],
        "chase": ["pursuit", "run chase", "target hunt"]
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
        retrieved_info = retrieval_model.retrieve(user_query, top_k=3)
        if not retrieved_info:
            return {"answer": "I'm not sure about that policy."}

        # Combine retrieved sections for better context
        context = " ".join(retrieved_info)

        # Create a prompt with context for the generative model
        prompt = (
            f"You are an assistant that provides detailed answers based on provided context.\n\n"
            f"Context: {context}\n"
            f"Question: {user_query}\n"
            "Answer:"
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
