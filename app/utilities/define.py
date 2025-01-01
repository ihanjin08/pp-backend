def define(strand, definitions):
    for term, definition in definitions.items():
        strand = strand.replace(term.lower(), f"{term.lower()} ({definition})")
    return strand