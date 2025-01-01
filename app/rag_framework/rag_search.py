import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from utilities.define import define
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def bm25_rag_search(chunks, embeddings, subject, criterion, strand, data):
    # Step 1: BM25 Retrieval
    query = data[subject][criterion]["Descriptors"][strand]
    query = define(query)

    tokenized_chunks = [chunk.split() for chunk in chunks]
    tokenized_query = query.split()

    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked_indices_bm25 = np.argsort(bm25_scores)[::-1]
    top_k = 15
    top_chunks_bm25 = [chunks[i] for i in ranked_indices_bm25[:top_k]]
    top_embeddings_bm25 = [embeddings[i] for i in ranked_indices_bm25[:top_k]]

    # Step 2: Compute Query Embedding
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings_model.embed_query(query)

    # Step 3: Re-rank with Cosine Similarity
    similarity_scores = cosine_similarity([query_embedding], top_embeddings_bm25)[0]
    ranked_indices_cosine = np.argsort(similarity_scores)[::-1]

    # Step 4: Retrieve Final Ranked Chunks
    final_top_chunks = [top_chunks_bm25[i] for i in ranked_indices_cosine]

    return final_top_chunks