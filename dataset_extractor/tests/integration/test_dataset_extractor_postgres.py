import os

import pandas as pd
import pytest
from sqlalchemy.engine import make_url

from main import ContextDatasetGenerator


@pytest.fixture

def generator():
    dsn = os.getenv("DATASET_EXTRACTOR_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Configure DATASET_EXTRACTOR_TEST_DATABASE_URL for PostgreSQL integration tests")

    url = make_url(dsn)
    if url.get_backend_name() != "postgresql":
        pytest.fail("Integration tests require a dedicated PostgreSQL test database")

    config = {
        "username": url.username,
        "password": url.password,
        "host": url.host,
        "port": url.port or 5432,
        "database": url.database,
    }
    instance = ContextDatasetGenerator(config)
    assert instance.connect()

    # This suite owns the context_records table in the configured test database.
    with instance.engine.begin() as connection:
        connection.exec_driver_sql("DROP VIEW IF EXISTS context_records_labels")
        connection.exec_driver_sql("DROP VIEW IF EXISTS context_records_ml")
        connection.exec_driver_sql("DROP TABLE IF EXISTS context_records")
        connection.exec_driver_sql(
            """
            CREATE TABLE context_records (
                id BIGINT PRIMARY KEY,
                request_id TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                risk_json JSONB NOT NULL,
                confidentiality_json JSONB NOT NULL,
                source_json JSONB NOT NULL,
                destination_json JSONB NOT NULL,
                headers_json JSONB
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO context_records VALUES (
                1, 'req-integration', '2026-01-02T03:04:05Z',
                '{"score": 0.1, "level": "low", "policy_overrides": []}',
                '{"score": 0.2, "classification": "public", "tags": [], "detected_patterns": []}',
                '{"ip": "192.168.1.10", "mfa_status": "enabled", "geo": "BR"}',
                '{"service_type": "api", "security_policy": "low", "allowed_protocols": ["TLS1.3"]}',
                '{"content-type": "application/json"}'
            )
            """
        )

    yield instance

    with instance.engine.begin() as connection:
        connection.exec_driver_sql("DROP VIEW IF EXISTS context_records_labels")
        connection.exec_driver_sql("DROP VIEW IF EXISTS context_records_ml")
        connection.exec_driver_sql("DROP TABLE IF EXISTS context_records")
    instance.engine.dispose()


def test_postgres_views_and_export_produce_expected_labels(generator, tmp_path):
    assert generator.drop_views()
    assert generator.create_ml_view()
    assert generator.create_labels_view()

    with generator.engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT risk_score, src_ip_private, dst_tls13_allowed, "
            "security_level_label, encryption_script_label, processing_priority_label "
            "FROM context_records_labels WHERE id = 1"
        ).mappings().one()

    assert row["risk_score"] == pytest.approx(0.1)
    assert row["src_ip_private"] is True
    assert row["dst_tls13_allowed"] is True
    assert row["security_level_label"] == "low"
    assert row["encryption_script_label"] == "aes128_gcm_tls13"
    assert row["processing_priority_label"] == "low_priority"

    output = tmp_path / "dataset.csv"
    frame = generator.export_dataset(str(output), sample_size=1)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.iloc[0]["security_level_label"] == "low"
    assert pd.read_csv(output).iloc[0]["encryption_script_label"] == "aes128_gcm_tls13"
