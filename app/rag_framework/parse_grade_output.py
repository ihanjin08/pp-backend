import re

def parse_grade_output(output: str):
    """
    Parses the output of the grade_strand method into JSON fields.

    Args:
        output (str): The raw output from the grade_strand method.

    Returns:
        dict: A dictionary with keys `strand`, `working_level`, `evidence`, and `reasoning`.
    """
    strand_pattern = r"Strand \*(.*?)\*: (.+)"
    working_level_pattern = r"Working Level: \((.*?)\)"
    evidence_pattern = r"Specific Evidence: (.+)"
    reasoning_pattern = r"Reasoning: (.+)"
    
    strand_match = re.search(strand_pattern, output)
    working_level_match = re.search(working_level_pattern, output)
    evidence_match = re.search(evidence_pattern, output)
    reasoning_match = re.search(reasoning_pattern, output)
    
    if not (strand_match and working_level_match and evidence_match and reasoning_match):
        raise ValueError("Output format does not match the expected strict format.")
    
    # Extract evidence as a list
    evidence = [e.strip('" ') for e in evidence_match.group(1).split('",') if e.strip()]
    
    return {
        "strand": f"Strand {strand_match.group(1)}: {strand_match.group(2)}",
        "working_level": working_level_match.group(1),
        "evidence": evidence,
        "reasoning": reasoning_match.group(1)
    }