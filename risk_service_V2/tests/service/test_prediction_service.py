import unittest
from unittest.mock import Mock, patch

from src.api.services.prediction_service import PredictionService


class PredictionServiceTests(unittest.TestCase):
    def test_initialization_loads_model_and_predict_forwards_arguments(self):
        manager = Mock()
        manager.predict.return_value = {"n_records": 1}

        service = PredictionService(manager)
        result = service.predict([{"amount": 10}], version="v2", models=["forest"])

        manager.load.assert_any_call()
        manager.load.assert_called_with("v2")
        manager.predict.assert_called_once_with([{"amount": 10}], model_names=["forest"])
        self.assertEqual(result, {"n_records": 1})

    def test_predict_without_version_does_not_reload(self):
        manager = Mock()
        service = PredictionService(manager)
        manager.load.reset_mock()

        service.predict([{"amount": 5}])

        manager.load.assert_not_called()
        manager.predict.assert_called_once_with([{"amount": 5}], model_names=None)

    def test_version_load_error_is_propagated_without_predicting(self):
        manager = Mock()
        service = PredictionService(manager)
        manager.load.reset_mock()
        manager.load.side_effect = FileNotFoundError("version absent")

        with patch("src.api.services.prediction_service.logger.exception"):
            with self.assertRaisesRegex(FileNotFoundError, "version absent"):
                service.predict([{"amount": 5}], version="missing")

        manager.load.assert_called_once_with("missing")
        manager.predict.assert_not_called()

    def test_initial_model_load_failure_is_logged_but_does_not_block_creation(self):
        manager = Mock()
        manager.load.side_effect = RuntimeError("models unavailable")

        with patch("src.api.services.prediction_service.logger.exception"):
            service = PredictionService(manager)

        self.assertIs(service.manager, manager)
        manager.predict.return_value = {"n_records": 0}
        self.assertEqual(service.predict([]), {"n_records": 0})

    def test_prediction_error_is_propagated_from_manager(self):
        manager = Mock()
        service = PredictionService(manager)
        manager.predict.side_effect = RuntimeError("prediction failed")

        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            service.predict([{"amount": 5}])


if __name__ == "__main__":
    unittest.main()
