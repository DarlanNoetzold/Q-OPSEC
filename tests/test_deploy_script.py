import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/deploy.sh"
GUIDE = ROOT / "docs/kb/03-deploy.md"


def test_script_and_guide_exist():
    assert SCRIPT.is_file(), "deploy.sh ausente"
    assert GUIDE.is_file(), "guia de deploy ausente"


def test_script_is_executable():
    assert SCRIPT.stat().st_mode & 0o111, "deploy.sh precisa ser executável"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_script_has_valid_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_script_rejects_unknown_target():
    result = subprocess.run(["bash", str(SCRIPT), "nao-existe"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "uso:" in result.stderr


def test_script_exposes_all_targets():
    content = SCRIPT.read_text(encoding="utf-8")
    for target in ("local", "compose", "k8s", "status", "verify"):
        assert target in content, f"alvo {target} não exposto no script"


def test_script_never_hardcodes_secrets():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "SECRETS_FILE" in content
    for leaked in ("replpass", "postgres:postgres@"):
        assert leaked not in content, f"credencial embutida no script: {leaked}"


def test_guide_covers_every_target():
    content = GUIDE.read_text(encoding="utf-8")
    for target in ("deploy.sh local", "deploy.sh compose", "deploy.sh k8s", "deploy.sh verify"):
        assert target in content, f"guia não cita {target}"
    assert "rollout undo" in content, "guia sem procedimento de rollback"
