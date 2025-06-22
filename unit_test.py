import unittest
import pandas as pd
from main import build_hash_index, hash_index_search, linear_search

class TestSearchFunctions(unittest.TestCase):
    def setUp(self):
        # Create a small DataFrame for testing
        data = {
            'Product ID': [1, 2, 3, 4, 5],
            'Name': ['A', 'B', 'C', 'D', 'E']
        }
        self.df = pd.DataFrame(data)
        self.hash_index = build_hash_index(self.df)

    def test_hash_index_search_found(self):
        ids = [1, 3, 5]
        # Should not raise and should complete
        try:
            hash_index_search(self.hash_index, ids)
        except Exception as e:
            self.fail(f"hash_index_search raised {e}")

    def test_hash_index_search_not_found(self):
        ids = [10, 20]
        # Should not raise and should complete
        try:
            hash_index_search(self.hash_index, ids)
        except Exception as e:
            self.fail(f"hash_index_search raised {e}")

    def test_linear_search_found(self):
        ids = [2, 4]
        # Should not raise and should complete
        try:
            linear_search(self.df, ids)
        except Exception as e:
            self.fail(f"linear_search raised {e}")

    def test_linear_search_not_found(self):
        ids = [99, 100]
        # Should not raise and should complete
        try:
            linear_search(self.df, ids)
        except Exception as e:
            self.fail(f"linear_search raised {e}")

if __name__ == '__main__':
    unittest.main()
