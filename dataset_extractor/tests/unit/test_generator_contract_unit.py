import pytest
from sqlalchemy.engine import make_url

from main import ContextDatasetGenerator


BASE_CONFIG = {
    "username": "user@tenant",
    "password": "p@ss/word:#",
    "host": "localhost",
    "port": 5432,
    "database": "context/db",
}


def test_connection_string_round_trips_reserved_characters():
    generator = ContextDatasetGenerator(BASE_CONFIG)

    parsed = make_url(generator.connection_string)

    assert parsed.username == BASE_CONFIG["username"]
    assert parsed.password == BASE_CONFIG["password"]
    assert parsed.host == BASE_CONFIG["host"]
    assert parsed.port == BASE_CONFIG["port"]
    assert parsed.database == BASE_CONFIG["database"]


@pytest.mark.parametrize("missing_key", ["username", "password", "host", "port", "database"])
def test_constructor_rejects_incomplete_database_configuration(missing_key):
    config = BASE_CONFIG.copy()
    del config[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        ContextDatasetGenerator(config)


def test_generator_starts_without_an_open_engine():
    generator = ContextDatasetGenerator(BASE_CONFIG)

    assert generator.engine is None
