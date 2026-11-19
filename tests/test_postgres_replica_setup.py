import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/postgres-replica/setup.sh"
README = ROOT / "deploy/postgres-replica/README.md"


def test_script_and_readme_exist():
    assert SCRIPT.is_file(), "setup.sh do pgBouncer/réplica ausente"
    assert README.is_file(), "README do pgBouncer/réplica ausente"


def test_script_is_executable():
    assert SCRIPT.stat().st_mode & 0o111, "setup.sh precisa ser executável"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_script_has_valid_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_script_rejects_unknown_argument():
    result = subprocess.run(["bash", str(SCRIPT), "--nao-existe"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "argumento desconhecido" in result.stderr


def test_script_documents_the_three_modes():
    content = SCRIPT.read_text(encoding="utf-8")
    for mode in ("--check", "--teardown", "apply"):
        assert mode in content, f"modo {mode} não documentado no script"
