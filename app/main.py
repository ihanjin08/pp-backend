from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List, Dict
from app.utilities.json_utils import load_json
from app.rag_framework.chunk import chunk_markdown
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.rag_framework.embed import embed
from app.rag_framework.rag_search import bm25_rag_search
from app.rag_framework.grade_strand import grade_strand
from app.rag_framework.final_grade import final_grade
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ibgrader.com"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


class Assignment(BaseModel):
    subject: Literal["Arts", "Design", "Individuals and Societies", "Language and Literature", "Mathematics", "Physical and Health Education", "Sciences"]
    criterion: Literal["A", "B", "C", "D"]
    content: str
    chunk_size: int | None = 250
    chunk_overlap: int | None = 50


class GradingResponse(BaseModel):
    feedback: List[Dict[str, str]]  # List of feedback for each strand
    final: int


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/grade", response_model=GradingResponse)
async def grade(input: Assignment):
    # Validate content type
    if not isinstance(input.content, str):
        raise ValueError("The content should be a valid string.")

    strands_data = load_json('resources/myp_subject_strands.json')
    chunks = chunk_markdown(input.content, input.chunk_size, input.chunk_overlap)
    embeddings = embed(chunks)
    feedback = []

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for i, strand in enumerate(strands_data[input.subject][input.criterion]["Descriptors"]):
        ranked_chunks = bm25_rag_search(chunks, embeddings, input.subject, input.criterion, i, strands_data)
        strand_feedback = grade_strand(ranked_chunks, input.subject, input.criterion, i, client, strands_data)
        feedback.append({"strand": i + 1, "feedback": strand_feedback})

    final = final_grade(feedback, client, input.criterion, input.subject)

    return {"feedback": feedback, "final": int(final)}