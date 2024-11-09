import json
import numpy as np
from retrieval_model import RetrievalModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Load dataset from JSON
with open("policy_query_dataset.json", "r") as f:
    dataset = json.load(f)


# Define evaluation functions
def precision_at_k(retrieved_docs, relevant_docs, k):
    relevant_retrieved = [
        doc for doc in retrieved_docs[:k] if any(rel_doc in doc for rel_doc in relevant_docs)
    ]
    return len(relevant_retrieved) / k if k > 0 else 0


def recall_at_k(retrieved_docs, relevant_docs, k):
    relevant_retrieved = [
        rel_doc for rel_doc in relevant_docs if any(rel_doc in doc for doc in retrieved_docs[:k])
    ]
    return len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0


def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    for i, doc in enumerate(retrieved_docs):
        if any(rel_doc in doc for rel_doc in relevant_docs):  # Partial matching
            return 1 / (i + 1)
    return 0


rouge = Rouge()


def bleu_score(generated_answer, ground_truth_answer):
    return sentence_bleu([ground_truth_answer.split()], generated_answer.split())


def rouge_score(generated_answer, ground_truth_answer):
    scores = rouge.get_scores(generated_answer, ground_truth_answer)
    return scores[0]['rouge-l']['f']


# Parameters
k = 5  # Top-k documents for retrieval

# List of retrieval and generative models to benchmark
retrieval_models = ["msmarco-distilbert-base-v4", "all-mpnet-base-v2", "distilbert-base-nli-stsb-mean-tokens"]
generative_models = ["google/flan-t5-small", "google/flan-t5-large"]

# Store results for each model combination
benchmark_results = []

# Begin benchmarking
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for retrieval_model_name in retrieval_models:
    # Initialize retrieval model
    print(f"\nEvaluating retrieval model: {retrieval_model_name}")
    retrieval_model = RetrievalModel(embedding_model_name=retrieval_model_name, structured_dir="documents_structured")

    for generative_model_name in generative_models:
        print(f"Evaluating generative model: {generative_model_name}")

        # Initialize generative model
        tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
        generative_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model_name).to(device)

        # Track metrics
        retrieval_metrics = []
        generation_metrics = []

        # Process each query in the dataset
        for entry in dataset:
            query = entry["query"]
            relevant_docs = set(entry["relevant_sections"])
            ground_truth_answer = entry["ground_truth_answer"]

            # Step 1: Retrieve top-k documents
            retrieved_docs = retrieval_model.retrieve(query, top_k=k)

            # Step 2: Calculate retrieval metrics
            precision = precision_at_k(retrieved_docs, relevant_docs, k)
            recall = recall_at_k(retrieved_docs, relevant_docs, k)
            mrr = mean_reciprocal_rank(retrieved_docs, relevant_docs)
            retrieval_metrics.append({"precision": precision, "recall": recall, "mrr": mrr})

            # Step 3: Generate answer using retrieved context
            context = " ".join(retrieved_docs)
            prompt = (
                "You are an assistant that provides detailed answers based on company policies.\n\n"
                f"Question: {query}\n"
                f"Policy Context: {context}\n"
                "Please provide a detailed answer based on the policy information provided."
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            output = generative_model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)

            # Step 4: Calculate generative metrics
            bleu = bleu_score(generated_answer, ground_truth_answer)
            rouge_l = rouge_score(generated_answer, ground_truth_answer)
            generation_metrics.append({"bleu": bleu, "rouge-l": rouge_l})

        # Summarize results for this model combination
        avg_precision = np.mean([metric["precision"] for metric in retrieval_metrics])
        avg_recall = np.mean([metric["recall"] for metric in retrieval_metrics])
        avg_mrr = np.mean([metric["mrr"] for metric in retrieval_metrics])
        avg_bleu = np.mean([metric["bleu"] for metric in generation_metrics])
        avg_rouge_l = np.mean([metric["rouge-l"] for metric in generation_metrics])

        # Store results
        benchmark_results.append({
            "retrieval_model": retrieval_model_name,
            "generative_model": generative_model_name,
            "precision@5": avg_precision,
            "recall@5": avg_recall,
            "mrr": avg_mrr,
            "bleu": avg_bleu,
            "rouge-l": avg_rouge_l
        })

# Print benchmark results
print("\n=== Benchmark Results ===")
for result in benchmark_results:
    print(f"Retrieval Model: {result['retrieval_model']}, Generative Model: {result['generative_model']}")
    print(f"Precision@5: {result['precision@5']:.2f}, Recall@5: {result['recall@5']:.2f}, MRR: {result['mrr']:.2f}")
    print(f"BLEU Score: {result['bleu']:.2f}, ROUGE-L Score: {result['rouge-l']:.2f}\n")
