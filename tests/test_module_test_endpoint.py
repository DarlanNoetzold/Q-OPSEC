import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import orchestrator_linux as orchestrator


class ModuleTestEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(orchestrator.APP)
        self.original_config = orchestrator.CONFIG
        self.original_state = orchestrator.STATE
        orchestrator.CONFIG = {"services": {"context_api": {"type": "spring", "cwd": "./context_api"}}}
        orchestrator.STATE = {}

    def tearDown(self):
        orchestrator.CONFIG = self.original_config
        orchestrator.STATE = self.original_state

    def test_run_and_poll_returns_complete_success_result(self):
        async def fake_create_subprocess_exec(*command, **kwargs):
            class CompletedProcess:
                returncode = 0
                async def wait(self):
                    return 1 if "--version" in command else 0
                async def communicate(self):
                    return (b"Tests run: 4, Failures: 0, Errors: 0, Skipped: 0\nBUILD SUCCESS", b"")
            return CompletedProcess()

        with patch.object(orchestrator, "test_python_interpreter", return_value="/usr/bin/python3"), \
             patch.object(orchestrator.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            start = self.client.post("/services/context_api/tests")
            self.assertEqual(202, start.status_code)
            run_id = start.json()["run_id"]
            result = self.client.get(f"/services/context_api/tests/{run_id}")

        self.assertEqual(200, result.status_code)
        self.assertEqual("passed", result.json()["status"])
        self.assertIn("Tests run: 4", result.json()["summary"])
        self.assertIn("BUILD SUCCESS", result.json()["output"])

    def test_zero_exit_without_test_summary_is_not_reported_as_passed(self):
        async def fake_create_subprocess_exec(*command, **kwargs):
            class CompletedProcess:
                returncode = 0
                async def wait(self):
                    return 1 if "--version" in command else 0
                async def communicate(self):
                    return (b"BUILD SUCCESS but no tests were discovered", b"")
            return CompletedProcess()

        with patch.object(orchestrator, "test_python_interpreter", return_value="/usr/bin/python3"), \
             patch.object(orchestrator.asyncio, "create_subprocess_exec", side_effect=fake_create_subprocess_exec):
            started = self.client.post("/services/context_api/tests").json()
            result = self.client.get(f"/services/context_api/tests/{started['run_id']}")

        self.assertEqual("failed", result.json()["status"])

    def test_run_rejects_modules_without_tests(self):
        with patch.object(orchestrator, "module_has_tests", return_value=False):
            response = self.client.post("/services/context_api/tests")
        self.assertEqual(400, response.status_code)
        self.assertIn("No tests", response.json()["detail"])

    def test_poll_rejects_a_run_belonging_to_another_module(self):
        orchestrator.STATE["__test_runs__"] = {
            "existing": {"run_id": "existing", "service": "other_module", "status": "passed"}
        }
        response = self.client.get("/services/context_api/tests/existing")
        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
