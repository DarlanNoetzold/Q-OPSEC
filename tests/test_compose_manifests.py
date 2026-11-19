"""Validação estática do docker-compose do Q-OPSEC.

Cobre o que quebra um `docker compose up`: imagem sem tag fixa, serviço sem
healthcheck, dependência apontando para serviço inexistente, porta publicada
fora de 127.0.0.1 e variável obrigatória não declarada.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "deploy/docker-compose.yml"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose):
    return compose["services"]


def test_compose_exists_and_parses(compose):
    assert compose.get("name") == "qopsec"
    assert compose.get("services"), "compose sem serviços"


def test_images_are_pinned(services):
    for name, svc in services.items():
        image = svc.get("image")
        if not image:
            continue
        assert ":" in image, f"{name}: imagem sem tag ({image})"
        assert not image.endswith(":latest"), f"{name}: usa :latest ({image})"


def test_services_have_healthcheck(services):
    for name, svc in services.items():
        assert svc.get("healthcheck"), f"{name}: sem healthcheck"


def test_depends_on_targets_exist(services):
    for name, svc in services.items():
        depends = svc.get("depends_on") or {}
        targets = depends if isinstance(depends, list) else list(depends)
        for target in targets:
            assert target in services, f"{name}: depends_on {target} inexistente"


def test_published_ports_bind_loopback(services):
    for name, svc in services.items():
        for port in svc.get("ports", []) or []:
            assert str(port).startswith("127.0.0.1:"), f"{name}: porta {port} exposta fora do loopback"


def test_required_secrets_are_declared(services):
    redis = services["redis"]["environment"]["REDIS_PASSWORD"]
    assert "QOPSEC_REDIS_PASSWORD" in redis
    rabbit = services["rabbitmq"]["environment"]
    assert "QOPSEC_RABBITMQ_USER" in rabbit["RABBITMQ_DEFAULT_USER"]
    assert "QOPSEC_RABBITMQ_PASSWORD" in rabbit["RABBITMQ_DEFAULT_PASS"]


def test_orchestrator_uses_shared_redis_and_rabbit(services):
    env = services["orchestrator"]["environment"]
    assert "redis:" in env["QOPSEC_REDIS_URL"]
    assert "rabbitmq:" in env["QOPSEC_RABBITMQ_URL"]
    assert env["QOPSEC_RATE_LIMIT_REQUESTS"]
    assert env["QOPSEC_PIPELINE_IDEMPOTENCY_TTL_SECONDS"]


def test_kms_uses_redis_backend(services):
    env = services["kms"]["environment"]
    assert env["QOPSEC_KMS_SESSION_BACKEND"] == "redis"
    assert "redis:" in env["QOPSEC_REDIS_URL"]


def test_named_volumes_are_declared(compose, services):
    declared = set(compose.get("volumes") or {})
    for name, svc in services.items():
        for mount in svc.get("volumes", []) or []:
            source = str(mount).split(":")[0]
            if source.startswith(".") or source.startswith("/"):
                continue
            assert source in declared, f"{name}: volume nomeado {source} não declarado"
