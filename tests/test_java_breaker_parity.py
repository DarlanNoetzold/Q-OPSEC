from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

BREAKER_FILES = {
    "interceptor_api": ROOT
    / "interceptor_api/src/main/java/tech/noetzold/interceptor_api/resilience/CircuitBreaker.java",
    "context_api": ROOT
    / "context_api/src/main/java/tech/noetzold/context_api/resilience/CircuitBreaker.java",
    "validation_send_api": ROOT
    / "validation_send_api/src/main/java/tech/noetzold/validation_send_api/resilience/CircuitBreaker.java",
}


def normalized(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("package "):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def test_breaker_copies_exist():
    missing = [name for name, path in BREAKER_FILES.items() if not path.is_file()]
    assert not missing, f"CircuitBreaker.java ausente em: {missing}"


def test_breaker_copies_do_not_diverge():
    reference_name, reference_path = next(iter(BREAKER_FILES.items()))
    reference = normalized(reference_path)
    diverged = {
        name: path
        for name, path in BREAKER_FILES.items()
        if name != reference_name and normalized(path) != reference
    }
    assert not diverged, (
        "CircuitBreaker.java divergiu de "
        f"{reference_name}: {sorted(diverged)}. Sincronize as copias ou extraia um modulo comum."
    )
