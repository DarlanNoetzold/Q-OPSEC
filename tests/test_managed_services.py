from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch
import asyncio
import yaml
import orchestrator_linux as orchestrator


class ManagedDockerServiceTests(IsolatedAsyncioTestCase):
    def test_monitoring_services_are_declared_in_startup_order(self):
        config_path = Path(orchestrator.BASE_DIR) / "services.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual("qopsec_redis", config["startup_order"][0])
        self.assertEqual("qopsec_rabbitmq", config["startup_order"][1])
        self.assertEqual("qopsec_prometheus", config["startup_order"][2])
        self.assertIn("com.qopsec.managed-by=orchestrator", config["services"]["qopsec_redis"]["labels"])
        self.assertIn("com.qopsec.managed-by=orchestrator", config["services"]["qopsec_rabbitmq"]["labels"])

    def test_env_placeholders_are_resolved_before_container_start(self):
        config = {
            "container_name": "qopsec-redis-test",
            "image": "redis:7.4.3",
            "ports": ["127.0.0.1:6380:6380"],
            "labels": ["com.qopsec.managed-by=orchestrator"],
            "env": {"REDIS_PASSWORD": "${TEST_REDIS_SECRET}"},
            "volumes": ["logs/qopsec-redis:/data"],
        }
        captured = {}

        class Process:
            returncode = 0

            async def communicate(self):
                return (b"container-id\n", b"")

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            return Process()

        with patch.dict("os.environ", {"TEST_REDIS_SECRET": "example-secret"}), \
             patch.object(orchestrator, "docker_container_exists", return_value=False), \
             patch.object(orchestrator, "docker_image_exists", return_value=True), \
             patch.object(orchestrator, "docker_running", return_value=True), \
             patch.object(orchestrator.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            result = asyncio.run(orchestrator.docker_start_container(config))

        self.assertEqual("started", result["status"])
        self.assertIn("REDIS_PASSWORD=example-secret", captured["args"])
        self.assertIn("qopsec-internal", captured["args"])

    async def test_rejects_non_localhost_port_bindings_for_shared_cache_and_broker(self):
        config = {
            "container_name": "qopsec-redis-test",
            "image": "redis:7.4.3",
            "ports": ["0.0.0.0:6380:6380"],
            "labels": ["com.qopsec.managed-by=orchestrator"],
            "volumes": ["logs/qopsec-redis:/data"],
        }
        with patch.object(orchestrator, "docker_container_exists", return_value=False), \
             patch.object(orchestrator, "docker_image_exists", return_value=True), \
             patch.object(orchestrator, "docker_running", return_value=True), \
             patch.object(orchestrator.asyncio, "create_subprocess_exec") as run:
            result = await orchestrator.docker_start_container(config)
            self.assertEqual("error", result["status"])
            self.assertIn("localhost", result["error"])
            run.assert_not_awaited()

    async def test_refuses_to_start_dependency_when_password_placeholder_missing(self):
        config = {
            "container_name": "qopsec-redis-test",
            "image": "redis:7.4.3",
            "labels": ["com.qopsec.managed-by=orchestrator"],
            "env": {"REDIS_PASSWORD": "${MISSING_TEST_SECRET}"},
        }
        with patch("os.environ", {}), \
             patch.object(orchestrator, "docker_container_exists", return_value=False), \
             patch.object(orchestrator, "docker_image_exists", return_value=True):
            result = await orchestrator.docker_start_container(config)
        self.assertEqual("error", result["status"])
        self.assertIn("MISSING_TEST_SECRET", result["error"])

    async def test_refuses_to_manage_unowned_existing_container(self):
        config = {
            "container_name": "qopsec-prom",
            "image": "prom/prometheus",
            "labels": [],
        }
        with patch.object(orchestrator, "docker_container_exists", return_value=True):
            result = await orchestrator.docker_start_container(config)
        self.assertEqual("error", result["status"])
        self.assertIn("unowned", result["error"])

    async def test_refuses_to_stop_unowned_existing_container(self):
        with patch.object(orchestrator, "docker_container_exists", return_value=True), \
             patch.object(orchestrator, "docker_container_labels", return_value=[]), \
             patch.object(orchestrator.subprocess, "check_call") as run:
            result = await orchestrator.docker_stop_container("qopsec-prom")
        self.assertEqual("error", result["status"])
        self.assertIn("unowned", result["error"])
        run.assert_not_called()

    def test_builds_docker_run_command_with_validated_read_only_mount_and_label(self):
        config = {
            "container_name": "qopsec-prom-test",
            "image": "prom/prometheus",
            "ports": ["19091:9090"],
            "volumes": ["monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro"],
            "labels": ["com.qopsec.managed-by=orchestrator"],
            "command": ["--config.file=/etc/prometheus/prometheus.yml"],
        }
        captured = {}

        class Process:
            returncode = 0

            async def communicate(self):
                return (b"container-id\n", b"")

        async def fake_create_subprocess_exec(*args, **kwargs):
            captured["args"] = args
            return Process()

        with patch.object(orchestrator, "docker_container_exists", return_value=False), \
             patch.object(orchestrator, "docker_image_exists", return_value=True), \
             patch.object(orchestrator, "docker_running", return_value=True), \
             patch.object(orchestrator.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            result = asyncio.run(orchestrator.docker_start_container(config))

        self.assertEqual("started", result["status"])
        self.assertIn("com.qopsec.managed-by=orchestrator", captured["args"])
        self.assertIn(f"{orchestrator.BASE_DIR}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro", captured["args"])
        self.assertNotIn("shell", str(captured["args"]))
