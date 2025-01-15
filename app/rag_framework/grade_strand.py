from openai import OpenAI
from app.utilities.define import define
import re

def grade_strand(context, subject, criterion, strand, client, data):
    """
    Grades a strand by analyzing the provided context and descriptors.

    Args:
        context (list): The ranked context chunks for the strand.
        subject (str): The subject of the assignment.
        criterion (str): The criterion being graded (e.g., "A", "B").
        strand (int): The index of the strand within the criterion.
        client (OpenAI): OpenAI client for generating completions.
        data (dict): JSON data containing subject, criterion, and descriptor information.

    Returns:
        dict: A structured dictionary containing strand information.
    """
    top_k = 5
    top_chunks_bm25 = ['"'+context[i]+'"' for i in range(min(top_k, len(context)))]
    context = "\n".join(top_chunks_bm25)

    # Build the level descriptors dynamically, only including strands that exist
    level_descriptors = []
    levels = ["1", "3", "5", "7"]
    for level in levels:
        if level in data[subject][criterion] and len(data[subject][criterion][level]) > strand:
            level_descriptors.append(f"\n{level}-2: {define(data[subject][criterion][level][strand])}\n")

    # Combine the descriptors for the prompt
    descriptors_section = "".join(level_descriptors)

    # Skip if no descriptors exist for the strand
    if not descriptors_section.strip():
        return {
            "strand": f"Strand {strand + 1}: {data[subject][criterion]['Descriptors'][strand]}",
            "feedback": "No descriptors available for this strand.",
            "evidence": [],
            "working_level": None,
            "reasoning": "No grading was possible due to lack of descriptors."
        }

    prompt = f"""
    You are part of an important IB MYP grading committee responsible for grading students' work based on specific evidence per strand.
    Here is the specific strand being assessed: \n{data[subject][criterion]["Descriptors"][strand]}

    Here is the specific evidence from the assignment:
    \n{context}\n
    Which of these descriptors best fits the level describes the student and give a specific example as to why. Give a working level (1-2, 3-4, 5-6, 7-8) and one piece of specific evidence.

    {descriptors_section}

    --FORMAT-- (STRICT REQUIREMENT)
    Strand *roman numeral*: (description)
    Working Level: (insert working level here, must choose from  (1-2, 3-4, 5-6, 7-8) based on evidence)
    Specific Evidence: "(insert < 1 sentence evidence here, insert as many pieces of evidence as necessary)", "(evidence #2, INCLUDE MORE THAN ONE IF AND ONLY IF RELEVANT AND HELPFUL)" ...
    Reasoning: (ensure that the reasoning is solid and dry, avoiding regurgitation of the command terms and MYP strand)
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o-mini",
    )

    response = chat_completion.choices[0].message.content

    # Parse and structure the output
    return {
        "strand": f"Strand {strand + 1}: {data[subject][criterion]['Descriptors'][strand]}",
        "working_level": extract_working_level(response),
        "evidence": extract_evidence(response),
        "reasoning": extract_reasoning(response)
    }


def extract_working_level(response: str):
    match = re.search(r"Working Level: \((.*?)\)", response)
    return match.group(1) if match else None


def extract_evidence(response: str):
    match = re.search(r"Specific Evidence: (.+)", response)
    if not match:
        return []
    return [e.strip('" ') for e in match.group(1).split('",') if e.strip()]


def extract_reasoning(response: str):
    match = re.search(r"Reasoning: (.+)", response)
    return match.group(1) if match else None
