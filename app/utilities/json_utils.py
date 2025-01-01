import json
from typing import Any

def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as file:
        return json.load(file)