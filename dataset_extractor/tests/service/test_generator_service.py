from unittest.mock import MagicMock, patch

import main
from main import ContextDatasetGenerator


CONFIG = {
    "username": "tester",
    "password": "secret",
    "host": "localhost",
    "port": 5432,
    "database": "contexts",
}


def test_connect_creates_engine_and_returns_true():
    generator = ContextDatasetGenerator(CONFIG)
    engine = object()

    with patch.object(main, "create_engine", return_value=engine) as create_engine:
        assert generator.connect() is True

    create_engine.assert_called_once_with(generator.connection_string)
    assert generator.engine is engine


def test_connect_maps_engine_error_to_false():
    generator = ContextDatasetGenerator(CONFIG)

    with patch.object(main, "create_engine", side_effect=RuntimeError("offline")):
        assert generator.connect() is False


def test_create_ml_view_executes_sql_and_commits():
    generator = ContextDatasetGenerator(CONFIG)
    connection = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    engine = MagicMock()
    engine.connect.return_value = connection_context
    generator.engine = engine

    assert generator.create_ml_view() is True

    connection.execute.assert_called_once()
    assert "CREATE OR REPLACE VIEW context_records_ml" in str(connection.execute.call_args.args[0])
    connection.commit.assert_called_once()


def test_view_creation_returns_false_when_database_rejects_sql():
    generator = ContextDatasetGenerator(CONFIG)
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("database unavailable")
    generator.engine = engine

    assert generator.create_labels_view() is False


def test_export_dataset_writes_csv_and_applies_requested_limit(tmp_path):
    generator = ContextDatasetGenerator(CONFIG)
    generator.engine = MagicMock()
    frame = MagicMock()
    frame.__len__.return_value = 1
    frame.columns = ["security_level_label", "encryption_script_label"]
    frame.__getitem__.return_value.value_counts.return_value = "counts"
    output = tmp_path / "sample.csv"

    with patch.object(main.pd, "read_sql", return_value=frame) as read_sql, patch("builtins.print"):
        assert generator.export_dataset(str(output), sample_size=7) is frame

    sql = read_sql.call_args.args[0]
    assert "ORDER BY created_at DESC" in sql
    assert sql.endswith("LIMIT 7")
    frame.to_csv.assert_called_once_with(str(output), index=False)


def test_export_dataset_returns_none_when_query_fails():
    generator = ContextDatasetGenerator(CONFIG)
    generator.engine = MagicMock()

    with patch.object(main.pd, "read_sql", side_effect=RuntimeError("query failed")):
        assert generator.export_dataset() is None
