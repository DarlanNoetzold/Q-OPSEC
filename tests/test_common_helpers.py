import os
import tempfile
import unittest

from common.http import SharedAsyncClient, parse_host_rewrites, rewrite_host
from common.ttl_cache import file_cache_key


class HostRewriteTests(unittest.TestCase):
    def test_parses_pairs_and_ignores_garbage(self):
        self.assertEqual({"a": "b", "c": "d"}, parse_host_rewrites(" a=b , c=d,broken,=x"))

    def test_rewrites_url_with_explicit_map(self):
        self.assertEqual("http://h2:1/x", rewrite_host("http://h1:1/x", {"h1": "h2"}))

    def test_env_map_overrides_default(self):
        old = os.environ.get("QOPSEC_HOST_REWRITES")
        os.environ["QOPSEC_HOST_REWRITES"] = ""
        try:
            self.assertEqual("http://10.0.0.5/x", rewrite_host("http://10.0.0.5/x"))
        finally:
            if old is None:
                os.environ.pop("QOPSEC_HOST_REWRITES")
            else:
                os.environ["QOPSEC_HOST_REWRITES"] = old


class SharedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_client_until_closed(self):
        shared = SharedAsyncClient(timeout=1.0)
        first = shared.get()
        self.assertIs(first, shared.get())
        await shared.aclose()
        self.assertTrue(first.is_closed)
        self.assertIsNot(first, shared.get())
        await shared.aclose()


class FileCacheKeyTests(unittest.TestCase):
    def test_key_changes_when_file_changes(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("a")
        try:
            before = file_cache_key(handle.name)
            with open(handle.name, "w") as again:
                again.write("abc")
            self.assertNotEqual(before, file_cache_key(handle.name))
        finally:
            os.unlink(handle.name)
