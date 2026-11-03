import subprocess
import sys
import unittest


class ModelTrainingPackageImportTests(unittest.TestCase):
    def test_package_import_does_not_load_training_only_dependencies(self):
        code = (
            "import sys; import src.model_training; "
            "assert 'src.model_training.trainer' not in sys.modules; "
            "assert 'imblearn' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_package_import_does_not_require_optional_frameworks(self):
        code = (
            "import sys; import src.model_training.models; "
            "assert 'lightgbm' not in sys.modules; "
            "assert 'xgboost' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
