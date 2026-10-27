from sqlalchemy.engine import URL

from main import ContextDatasetGenerator


def test_connection_url_escapes_credentials_and_database_name():
    generator = ContextDatasetGenerator(
        {
            "username": "user@tenant",
            "password": "p@ss/word:#",
            "host": "localhost",
            "port": 5432,
            "database": "context/db",
        }
    )

    assert generator.connection_string == URL.create(
        "postgresql+psycopg2",
        username="user@tenant",
        password="p@ss/word:#",
        host="localhost",
        port=5432,
        database="context/db",
    ).render_as_string(hide_password=False)
