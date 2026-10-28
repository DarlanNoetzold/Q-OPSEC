from unittest.mock import MagicMock, patch

import pandas as pd

import main
from main import ContextDatasetGenerator


CONFIG = {
    "username": "tester",
    "password": "secret",
    "host": "localhost",
    "port": 5432,
    "database": "contexts",
}


def make_generator():
    generator = ContextDatasetGenerator(CONFIG)
    generator.engine = MagicMock()
    return generator


def test_drop_views_executes_both_drops_and_commits():
    generator = make_generator()
    connection = MagicMock()
    generator.engine.connect.return_value.__enter__.return_value = connection

    assert generator.drop_views() is True

    statements = [str(call.args[0]) for call in connection.execute.call_args_list]
    assert len(statements) == 2
    assert "DROP VIEW IF EXISTS context_records_labels" in statements[0]
    assert "DROP VIEW IF EXISTS context_records_ml" in statements[1]
    connection.commit.assert_called_once()


def test_drop_views_returns_false_on_database_error():
    generator = make_generator()
    generator.engine.connect.side_effect = RuntimeError("database unavailable")

    assert generator.drop_views() is False


def test_create_labels_view_uses_ml_view_and_commits():
    generator = make_generator()
    connection = MagicMock()
    generator.engine.connect.return_value.__enter__.return_value = connection

    assert generator.create_labels_view() is True

    sql = str(connection.execute.call_args.args[0])
    assert "CREATE OR REPLACE VIEW context_records_labels" in sql
    assert "FROM context_records_ml" in sql
    assert "THEN 'critical'" in sql
    connection.commit.assert_called_once()


def test_get_dataset_info_returns_summary_and_distribution_frames():
    generator = make_generator()
    summary = pd.DataFrame([{"total_records": 2, "security_levels": 2,
                             "encryption_scripts": 2, "oldest_record": "old",
                             "newest_record": "new"}])
    distribution = pd.DataFrame([{"security_level_label": "low", "count": 2}])

    with patch.object(main.pd, "read_sql", side_effect=[summary, distribution]) as read_sql, \
            patch("builtins.print"):
        actual_summary, actual_distribution = generator.get_dataset_info()

    pd.testing.assert_frame_equal(actual_summary, summary)
    pd.testing.assert_frame_equal(actual_distribution, distribution)
    assert read_sql.call_count == 2
    assert "context_records_labels" in read_sql.call_args_list[0].args[0]
    assert "GROUP BY security_level_label, encryption_script_label" in read_sql.call_args_list[1].args[0]


def test_get_dataset_info_returns_none_pair_on_query_error():
    generator = make_generator()
    with patch.object(main.pd, "read_sql", side_effect=RuntimeError("query failed")):
        assert generator.get_dataset_info() == (None, None)


def test_feature_engineering_export_writes_csv_and_returns_frame(tmp_path):
    generator = make_generator()
    frame = pd.DataFrame([{"risk_score": 0.5, "hour_sin": 0.0}])
    output = tmp_path / "engineered.csv"

    with patch.object(main.pd, "read_sql", return_value=frame) as read_sql:
        result = generator.create_feature_engineering_dataset(str(output))

    pd.testing.assert_frame_equal(result, frame)
    assert output.exists()
    pd.testing.assert_frame_equal(pd.read_csv(output), frame)
    sql = read_sql.call_args.args[0]
    assert "risk_conf_interaction" in sql
    assert "request_id_resolved" in sql


def test_feature_engineering_export_returns_none_on_query_error():
    generator = make_generator()
    with patch.object(main.pd, "read_sql", side_effect=RuntimeError("query failed")):
        assert generator.create_feature_engineering_dataset() is None


def test_export_dataset_returns_none_when_csv_write_fails(tmp_path):
    generator = make_generator()
    frame = pd.DataFrame([{"security_level_label": "low",
                           "encryption_script_label": "aes128_gcm"}])

    with patch.object(main.pd, "read_sql", return_value=frame), \
            patch.object(frame, "to_csv", side_effect=OSError("disk full")), \
            patch("builtins.print"):
        assert generator.export_dataset(str(tmp_path / "dataset.csv")) is None
