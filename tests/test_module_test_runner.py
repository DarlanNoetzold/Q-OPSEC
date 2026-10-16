import tempfile
import unittest
from pathlib import Path

from orchestrator_linux import build_test_command, module_has_tests, resolve_module_cwd


class ModuleTestRunnerTests(unittest.TestCase):
    def test_spring_modules_use_maven_test_goal(self):
        self.assertEqual(
            ["mvn", "-B", "-Dstyle.color=never", "test"],
            build_test_command("spring", "python3", pytest_available=True),
        )

    def test_python_modules_use_pytest_when_available(self):
        self.assertEqual(
            ["python3", "-m", "pytest", "-q"],
            build_test_command("fastapi", "python3", pytest_available=True),
        )

    def test_python_modules_fall_back_to_unittest_without_pytest(self):
        self.assertEqual(
            ["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test*.py"],
            build_test_command("process", "python3", pytest_available=False),
        )
        self.assertEqual(
            ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"],
            build_test_command("fastapi", "python3", pytest_available=False, tests_directory=True),
        )

    def test_module_cwd_resolves_inside_project_root(self):
        root = Path("/tmp/qopsec-test-root")
        self.assertEqual(root / "context_api", resolve_module_cwd({"cwd": "./context_api"}, root))

    def test_module_cwd_rejects_paths_outside_project_root(self):
        root = Path("/tmp/qopsec-test-root")
        with self.assertRaises(ValueError):
            resolve_module_cwd({"cwd": "../../etc"}, root)

    def test_python_module_detection_ignores_virtualenv_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vendor_tests = root / "qopsec_env" / "lib" / "site-packages"
            vendor_tests.mkdir(parents=True)
            (vendor_tests / "test_dependency.py").touch()
            self.assertFalse(module_has_tests("fastapi", root))
            project_tests = root / "tests"
            project_tests.mkdir()
            (project_tests / "test_app.py").touch()
            self.assertTrue(module_has_tests("fastapi", root))


if __name__ == "__main__":
    unittest.main()
