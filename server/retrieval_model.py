from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)


class RetrievalModel:
    def __init__(self, embedding_model_name, structured_dir="documents_structured"):
        # Initialize the pre-trained retrieval model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load and embed all structured documents
        self.corpus_metadata, self.corpus_texts, self.corpus_embeddings = self.load_and_embed_structured_documents(
            structured_dir)

        # Ensure valid embeddings are available
        if len(self.corpus_embeddings) == 0 or self.corpus_embeddings.shape[1] == 0:
            raise ValueError("No valid embeddings found. Ensure the structured documents contain valid content.")

        # Create FAISS index for efficient retrieval
        self.index = self.create_faiss_index(self.corpus_embeddings)

    def load_and_embed_structured_documents(self, structured_dir):
        # Load all structured documents from the folder
        corpus = []
        metadata = []  # List to store metadata (file names and section IDs)

        for filename in os.listdir(structured_dir):
            if filename.endswith("_structured.txt"):
                file_path = os.path.join(structured_dir, filename)
                with open(file_path, 'r') as f:
                    document = f.read().split('\n\n')  # Split by double newlines for structured sections
                    for idx, section in enumerate(document):
                        section_text = section.strip()
                        if section_text:
                            corpus.append(section_text)
                            metadata.append(f"{filename} Section {idx + 1}")  # Store filename and section number

        # Check if the corpus is empty
        if not corpus:
            raise ValueError("No valid structured documents found in the directory.")

        # Encode the corpus for embeddings
        embeddings = self.embedding_model.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')  # Convert to float32 for FAISS

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        return metadata, corpus, embeddings

    def create_faiss_index(self, embeddings):
        dim = embeddings.shape[1]

        # Create a FAISS index using cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index

    # def retrieve(self, query, top_k=3):
    #     # Encode the query
    #     query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
    #     query_embedding = np.array(query_embedding).astype('float32')
    #     faiss.normalize_L2(query_embedding)
    #
    #     # Search the FAISS index for the top_k most relevant sections
    #     distances, indices = self.index.search(query_embedding, top_k)
    #
    #     # Retrieve the top_k relevant sections along with metadata
    #     retrieved_results = [
    #         f"{self.corpus_metadata[idx]}: {self.corpus_texts[idx]}"
    #         for idx in indices[0] if idx < len(self.corpus_texts)
    #     ]
    #     return retrieved_results

    def retrieve(self, query, top_k=3):
        """
        Retrieve top K relevant documents based on the query.
        """
        # Step 1: Encode the query into an embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Step 2: Search the FAISS index for the top_k most relevant sections
        distances, indices = self.index.search(query_embedding, top_k)

        # Step 3: Retrieve the top_k relevant sections along with metadata
        retrieved_results = [
            f"{self.corpus_metadata[idx]}: {self.corpus_texts[idx]}"
            for idx in indices[0] if idx < len(self.corpus_texts)
        ]

        # Step 4: Keyword filtering to improve relevance
        keywords = query.lower().split()  # Split query into keywords
        filtered_results = [
            result for result in retrieved_results
            if any(keyword in result.lower() for keyword in keywords)
        ]

        # If filtering results in an empty list, return the original retrieved results
        if not filtered_results:
            return retrieved_results

        return filtered_results

