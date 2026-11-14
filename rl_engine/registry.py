import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path: Path, data, **dump_kwargs) -> None:
    """Write to a temp file and rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class RLRegistry:
    def __init__(self, path: Path):
        self.path = path

    def save(self, q_table: dict):
        write_json_atomic(self.path, q_table)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, "r") as f:
            return json.load(f)
