import os
import unittest
from unittest import mock

from folium.context import estimate_tokens
from folium.token_estimator import estimate_text_tokens


class TokenEstimatorTests(unittest.TestCase):
    def test_default_estimator_falls_back_to_approx_without_tokenizer_path(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(estimate_text_tokens("abcdef"), 2)

    def test_deepseek_estimator_falls_back_without_tokenizer_path(self):
        with mock.patch.dict(os.environ, {"FOLIUM_TOKEN_ESTIMATOR": "deepseek"}, clear=True):
            self.assertEqual(estimate_text_tokens("abcdef"), 2)

    def test_default_deepseek_estimator_uses_configured_tokenizer(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1, 2, 3]

        with mock.patch.dict(os.environ, {"FOLIUM_DEEPSEEK_TOKENIZER": "D:\\tokenizers\\deepseek"}, clear=True):
            with mock.patch("folium.token_estimator._load_deepseek_tokenizer", return_value=tokenizer):
                self.assertEqual(estimate_text_tokens("abcdef"), 3)

    def test_deepseek_estimator_uses_configured_tokenizer(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4]

        with mock.patch.dict(
            os.environ,
            {
                "FOLIUM_TOKEN_ESTIMATOR": "deepseek",
                "FOLIUM_DEEPSEEK_TOKENIZER": "D:\\tokenizers\\deepseek",
            },
            clear=True,
        ):
            with mock.patch("folium.token_estimator._load_deepseek_tokenizer", return_value=tokenizer):
                self.assertEqual(estimate_text_tokens("abcdef"), 4)

    def test_estimate_tokens_uses_configured_estimator(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            messages = [{"role": "user", "content": "abcdef"}]

            self.assertEqual(estimate_tokens(messages), 2)


if __name__ == "__main__":
    unittest.main()
