import unittest

from common.ttl_cache import TTLCache


class TTLCacheTests(unittest.TestCase):
    def test_get_or_set_computes_value_only_once(self):
        cache = TTLCache()
        calls = []

        def factory():
            calls.append(1)
            return {"value": 1}

        self.assertEqual({"value": 1}, cache.get_or_set("key", factory, 30))
        self.assertEqual({"value": 1}, cache.get_or_set("key", factory, 30))
        self.assertEqual(1, len(calls))

    def test_entries_expire_after_ttl(self):
        now = [0.0]
        cache = TTLCache(clock=lambda: now[0])
        cache.set("key", "value", 5)

        now[0] = 6.0

        self.assertIsNone(cache.get("key"))

    def test_entries_are_bounded_and_evict_oldest(self):
        cache = TTLCache(max_entries=2)
        cache.set("a", 1, 30)
        cache.set("b", 2, 30)
        cache.set("c", 3, 30)

        self.assertIsNone(cache.get("a"))
        self.assertEqual(2, cache.get("b"))
        self.assertEqual(3, cache.get("c"))


if __name__ == "__main__":
    unittest.main()
