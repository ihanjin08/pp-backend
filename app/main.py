from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
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

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ibgrader.com"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Define request model
class Assignment(BaseModel):
    subject: Literal[
        "Arts",
        "Design",
        "Individuals and Societies",
        "Language and Literature",
        "Mathematics",
        "Physical and Health Education",
        "Sciences"
    ]
    criterion: Literal["A", "B", "C", "D"]
    content: str
    chunk_size: int | None = 250
    chunk_overlap: int | None = 50


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/grade")
async def grade(input: Assignment):
    """
    Endpoint to grade an assignment based on provided content, subject, and criterion.

    Args:
        input (Assignment): The input data for grading.

    Returns:
        dict: The grading results with feedback and a final grade.
    """
    # Load subject strand data
    strands_data = load_json('resources/myp_subject_strands.json')
    
    # Chunk the content and generate embeddings
    chunks = chunk_markdown(input.content, input.chunk_size, input.chunk_overlap)
    embeddings = embed(chunks)

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Process each strand and collect feedback
    feedback = []
    for i, descriptor in enumerate(strands_data[input.subject][input.criterion]["Descriptors"]):
        # Rank chunks using BM25 and grade each strand
        ranked_chunks = bm25_rag_search(chunks, embeddings, input.subject, input.criterion, i, strands_data)
        strand_feedback = grade_strand(ranked_chunks, input.subject, input.criterion, i, client, strands_data)
        feedback.append(strand_feedback)

    # Calculate the final grade
    final = final_grade([f["working_level"] for f in feedback if f["working_level"]], client, input.criterion, input.subject)

    return {
        "strands": feedback,
        "final": final
    }