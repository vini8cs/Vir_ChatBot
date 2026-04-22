"""Tests for llms/tokenizer.py (TokenizerWrapper).

AutoTokenizer.from_pretrained would otherwise download a 7B model —
we patch it to return a fake tokenizer whose methods we control.
"""

from unittest.mock import MagicMock, patch

import pytest

from llms.tokenizer import TokenizerWrapper


@pytest.fixture()
def fake_tokenizer():
    tk = MagicMock()
    tk.vocab_size = 32000
    tk.encode.return_value = [1, 2, 3, 4]
    tk.decode.return_value = "hello world"
    return tk


@pytest.fixture()
def wrapper(fake_tokenizer):
    with patch(
        "llms.tokenizer.AutoTokenizer.from_pretrained",
        return_value=fake_tokenizer,
    ):
        yield TokenizerWrapper(model_name="fake-model", max_length=128)


class TestTokenizerWrapper:
    def test_sets_model_max_length_on_underlying_tokenizer(
        self, fake_tokenizer
    ):
        with patch(
            "llms.tokenizer.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ):
            TokenizerWrapper(model_name="fake-model", max_length=64)
        assert fake_tokenizer.model_max_length == 64

    def test_get_tokenizer_returns_underlying(self, wrapper, fake_tokenizer):
        assert wrapper.get_tokenizer() is fake_tokenizer

    def test_get_max_tokens_returns_init_value(self, wrapper):
        assert wrapper.get_max_tokens() == 128

    def test_encode_delegates_without_special_tokens(
        self, wrapper, fake_tokenizer
    ):
        assert wrapper.encode("hi there") == [1, 2, 3, 4]
        fake_tokenizer.encode.assert_called_once_with(
            "hi there", add_special_tokens=False
        )

    def test_decode_delegates(self, wrapper, fake_tokenizer):
        assert wrapper.decode([1, 2, 3]) == "hello world"
        fake_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_count_tokens_returns_len_of_encoded(self, wrapper):
        assert wrapper.count_tokens("hi there") == 4

    def test_vocab_size_property(self, wrapper):
        assert wrapper.vocab_size == 32000

    def test_default_model_name_and_max_length(self, fake_tokenizer):
        with patch(
            "llms.tokenizer.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as mock_from_pretrained:
            w = TokenizerWrapper()
        mock_from_pretrained.assert_called_once_with(
            "mistralai/Mistral-7B-v0.1"
        )
        assert w.get_max_tokens() == 8191
