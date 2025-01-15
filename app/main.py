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
from app.rag_framework.parse_grade_output import parse_grade_output

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


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/grade")
async def grade(input: Assignment):
    strands_data = load_json('resources/myp_subject_strands.json')
    chunks = chunk_markdown(input.content, input.chunk_size, input.chunk_overlap)
    embeddings = embed(chunks)
    strands = []

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for i, descriptor in enumerate(strands_data[input.subject][input.criterion]["Descriptors"]):
        ranked_chunks = bm25_rag_search(chunks, embeddings, input.subject, input.criterion, i, strands_data)
        raw_output = grade_strand(ranked_chunks, input.subject, input.criterion, i, client, strands_data)
        
        # Parse the raw output from `grade_strand`
        try:
            parsed_output = parse_grade_output(raw_output)
            strands.append(parsed_output)
        except ValueError as e:
            strands.append({
                "strand": f"Strand {i + 1}: {descriptor}",
                "error": str(e)
            })

    final = final_grade([s["working_level"] for s in strands if "working_level" in s], client, input.criterion, input.subject)

    return {
        "strands": strands,
        "final": int(final)
    }