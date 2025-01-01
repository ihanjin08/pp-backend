from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from utilities.json_utils import load_json
from rag_framework.chunk import chunk_markdown
from openai import OpenAI
import os
from dotenv import load_dotenv
from rag_framework.embed import embed
from rag_framework.rag_search import bm25_rag_search
from rag_framework.grade_strand import grade_strand

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()

class Assignment(BaseModel):
    subject: Literal["Arts", "Design", "Individuals and Societies", "Language and Literature", "Mathematics", "Physical and Health Education", "Sciences"]
    criterion: Literal["A", "B", "C", "D"]
    content: str
    chunk_size: int | None = 250
    chunk_overlap: int | None = 50

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/grade")
async def grade(input: Assignment):
    strands_data = load_json('data/strands.json')
    chunks = chunk_markdown(Assignment.content, Assignment.chunk_size, Assignment.chunk_overlap)
    embeddings = embed(chunks)
    feedback = []

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for i, strand in enumerate(strands_data[Assignment.subject][Assignment.criterion]["Descriptors"], desc="Assessing strands"):
        ranked_chunks = bm25_rag_search(chunks, embeddings, Assignment.subject, Assignment.criterion, i)
        strand_feedback = grade_strand(ranked_chunks, Assignment.subject, Assignment.criterion, i, client)
        feedback.append(strand_feedback)
    
    return feedback