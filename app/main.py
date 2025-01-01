from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from utilities.json_utils import load_json
from utilities.define import define
from rag_framework.chunk import chunk_markdown

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
    definitions_data = load_json('data/myp_command_terms.json')
    chunks = chunk_markdown(Assignment.content, Assignment.chunk_size, Assignment.chunk_overlap)
    return