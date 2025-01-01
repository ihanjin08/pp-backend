from app.utilities.json_utils import load_json

def define(strand):
    definitions_data = load_json('resources/myp_command_terms.json')
    for term, definition in definitions_data.items():
        strand = strand.replace(term.lower(), f"{term.lower()} ({definition})")
    return strand