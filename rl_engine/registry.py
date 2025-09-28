import json
from pathlib import Path

class RLRegistry:
    def __init__(self, path: Path):
        self.path = path

    def save(self, q_table: dict):
        with open(self.path, "w") as f:
            json.dump(q_table, f)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "r") as f:
            return json.load(f)