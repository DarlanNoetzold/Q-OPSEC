import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import create_app
from app.api.v1 import endpoints_datasets
from app.core.security import get_current_user, require_auth


class DatasetsApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        root_patch = patch.object(endpoints_datasets, "DATASETS_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        endpoints_datasets._ANALYSIS_CACHE.clear()

        self.app = create_app()
        self.app.dependency_overrides[get_current_user] = lambda: {"user": "test"}
        self.app.dependency_overrides[require_auth] = lambda: {"user": "test"}
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

        (self.root / "ds1").mkdir()
        (self.root / "ds1" / "data.csv").write_text("score,label\n1,a\n2,b\n3,a\n")

    def test_list_and_get_dataset(self):
        listed = self.client.get("/api/v1/datasets").json()
        self.assertEqual(["ds1"], [d["name"] for d in listed])
        self.assertEqual(["data.csv"], listed[0]["files"])
        self.assertEqual(200, self.client.get("/api/v1/datasets/ds1").status_code)

    def test_schema_is_cached_until_file_changes(self):
        first = self.client.get("/api/v1/datasets/ds1/schema", params={"file": "data.csv"}).json()
        self.assertEqual([3, 2], first["shape"])
        with patch.object(endpoints_datasets, "_read_table", side_effect=AssertionError("cache miss")):
            cached = self.client.get("/api/v1/datasets/ds1/schema", params={"file": "data.csv"}).json()
        self.assertEqual(first, cached)

        (self.root / "ds1" / "data.csv").write_text("score,label\n1,a\n")
        changed = self.client.get("/api/v1/datasets/ds1/schema", params={"file": "data.csv"}).json()
        self.assertEqual([1, 2], changed["shape"])

    def test_stats_and_preview(self):
        stats = self.client.get("/api/v1/datasets/ds1/stats", params={"file": "data.csv"}).json()
        self.assertEqual({"a": 2, "b": 1}, stats["categorical_freqs"]["label"])
        preview = self.client.get("/api/v1/datasets/ds1/preview", params={"file": "data.csv", "n": 2}).json()
        self.assertEqual(2, preview["count_returned"])

    def test_traversal_is_rejected(self):
        response = self.client.get("/api/v1/datasets/ds1/schema", params={"file": "../../etc/passwd"})
        self.assertEqual(400, response.status_code)
        response = self.client.post("/api/v1/datasets", json={"name": "../escape"})
        self.assertEqual(400, response.status_code)

    def test_upload_strips_directory_from_filename(self):
        response = self.client.post(
            "/api/v1/datasets/ds1/files",
            files={"file": ("../../evil.csv", b"a,b\n1,2\n", "text/csv")},
        )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("evil.csv", response.json()["filename"])
        self.assertTrue((self.root / "ds1" / "evil.csv").exists())

    def test_delete_dataset_removes_tree(self):
        (self.root / "ds1" / "nested").mkdir()
        (self.root / "ds1" / "nested" / "x.txt").write_text("x")
        self.assertEqual(204, self.client.delete("/api/v1/datasets/ds1").status_code)
        self.assertFalse((self.root / "ds1").exists())
