import time
import unittest

from storage import KeySessionStore


class KeySessionStoreRegressionTests(unittest.TestCase):
    def test_replacing_session_removes_old_request_index(self):
        store = KeySessionStore()
        store.save("session-1", "request-old", "AES256_GCM", "key-1", int(time.time()) + 60)
        store.save("session-1", "request-new", "AES256_GCM", "key-2", int(time.time()) + 60)

        self.assertIsNone(store.get_by_request_id("request-old"))
        self.assertEqual("session-1", store.get_by_request_id("request-new")["session_id"])
        self.assertEqual(1, store.stats()["request_index_size"])

    def test_replacing_request_removes_old_session(self):
        store = KeySessionStore()
        store.save("session-1", "request-1", "AES256_GCM", "key-1", int(time.time()) + 60)
        store.save("session-2", "request-1", "AES256_GCM", "key-2", int(time.time()) + 60)

        self.assertIsNone(store.get_by_session_id("session-1"))
        self.assertEqual("session-2", store.get_by_request_id("request-1")["session_id"])
        self.assertEqual(1, store.stats()["total_sessions"])

    def test_request_lookup_runs_periodic_cleanup(self):
        store = KeySessionStore(cleanup_interval_seconds=0)
        store.save("session-expired", "request-expired", "AES256_GCM", "key", int(time.time()) - 1)

        self.assertIsNone(store.get_by_request_id("request-missing"))
        self.assertEqual(0, store.stats()["total_sessions"])
        self.assertEqual(0, store.stats()["request_index_size"])


if __name__ == "__main__":
    unittest.main()
