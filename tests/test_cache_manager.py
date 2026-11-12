"""
Unit tests for CAGCacheManager and DynamicCache manipulations.
"""

import unittest
from pathlib import Path
import tempfile
import sys
import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))

from cag_okf.cache_manager import CAGCacheManager
from cag_okf.models import CacheProfile


class TestCAGCacheManager(unittest.TestCase):

    def test_crop_and_clone(self):
        cache = DynamicCache()
        # Simulate 2 layers with dummy key/value tensors of seq_len 10
        k = torch.randn(1, 2, 10, 8)
        v = torch.randn(1, 2, 10, 8)
        cache.update(k, v, 0)

        self.assertEqual(cache.get_seq_length(), 10)

        # Clone
        cloned = CAGCacheManager.clone_cache(cache)
        self.assertEqual(cloned.get_seq_length(), 10)

        # Crop original
        CAGCacheManager.rollback_cache(cache, 4)
        self.assertEqual(cache.get_seq_length(), 4)
        # Cloned remains 10
        self.assertEqual(cloned.get_seq_length(), 10)

    def test_save_and_load_cache(self):
        cache = DynamicCache()
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        cache.update(k, v, 0)
        prefix_ids = torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108]])

        profile = CacheProfile(
            model_id="test-model",
            total_tokens=8,
            document_ids=["DOC-1"],
            created_at="2026-11-11T00:00:00",
            content_hash="abc123hash",
            prefix_prompt_preview="preview..."
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = str(Path(tmp_dir) / "test_cache.pt")
            CAGCacheManager.save_cache(cache, prefix_ids, profile, file_path)

            loaded_cache, loaded_ids, loaded_profile = CAGCacheManager.load_cache(file_path)
            self.assertEqual(loaded_cache.get_seq_length(), 8)
            self.assertEqual(loaded_ids.shape[1], 8)
            self.assertEqual(loaded_profile.model_id, "test-model")
            self.assertEqual(loaded_profile.content_hash, "abc123hash")


if __name__ == "__main__":
    unittest.main()
