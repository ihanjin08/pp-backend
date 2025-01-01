import logging
from openai import OpenAI
from app.utilities.define import define

def grade_strand(context, subject, criterion, strand, client, data):
    logging.info(f"Grading strand {strand + 1}.")

    top_k = 15
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
        logging.info(f"Skipping strand {strand + 1} as no descriptors exist across all levels.")
        return None

    prompt = f"""
    You are part of an important IB MYP grading committee responsible for grading students' work based on specific evidence per strand.
    Here is the specific strand being assessed: \n{data[subject][criterion]["Descriptors"][strand-1]}

    Here is the specific evidence from the assignment:
    \n{context}\n
    Which of these descriptors best fits the level describes the student and give a specific example as to why. Give a working level (1-2, 3-4, 5-6, 7-8) and one piece of specific evidence.

    {descriptors_section}

    --FORMAT-- (STRICT REQUIREMENT)
    Strand (input what strand it is for)
    
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

    return chat_completion.choices[0].message.content
