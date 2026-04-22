"""Tests for pure utilities in agents/vir_chatbot/vectorstore.py.

VectorStoreCreator.__init__ calls Gemini.__init__ (requires credentials)
and Docling. These tests only call @staticmethod methods and instantiate
the custom exception classes — no constructor is invoked.
"""

import pytest

from agents.vir_chatbot.vectorstore import (
    NoCacheFoundError,
    NoNewPDFError,
    NoVectorStoreFoundError,
    VectorAlreadyCreatedError,
    VectorStoreCreator,
)


class TestCleanText:
    # ── returns None for short / invalid input ─────────────────────────────

    def test_none_input_returns_none(self):
        assert VectorStoreCreator.clean_text(None) is None

    def test_non_string_returns_none(self):
        assert VectorStoreCreator.clean_text(123) is None

    def test_empty_string_returns_none(self):
        assert VectorStoreCreator.clean_text("") is None

    def test_three_words_returns_none(self):
        assert VectorStoreCreator.clean_text("one two three") is None

    # ── returns cleaned string for valid input ─────────────────────────────

    def test_four_words_returns_string(self):
        result = VectorStoreCreator.clean_text("one two three four")
        assert result == "one two three four"

    def test_strips_bullet_unicode_character(self):
        result = VectorStoreCreator.clean_text("\u2022 first point here now")
        assert result is not None
        assert "\u2022" not in result

    def test_collapses_internal_whitespace(self):
        result = VectorStoreCreator.clean_text(
            "word   extra   spaces   in text"
        )
        assert result == "word extra spaces in text"

    def test_strips_leading_and_trailing_whitespace(self):
        result = VectorStoreCreator.clean_text("  hello world foo bar  ")
        assert result == "hello world foo bar"

    def test_mixed_whitespace_and_bullet(self):
        result = VectorStoreCreator.clean_text(
            "  \u2022  term  definition  here  extra  "
        )
        assert result is not None
        assert "\u2022" not in result
        assert "  " not in result


class TestCustomExceptions:
    def test_no_new_pdf_error_has_default_message(self):
        exc = NoNewPDFError()
        assert "No new document" in str(exc)

    def test_no_new_pdf_error_accepts_custom_message(self):
        exc = NoNewPDFError("custom message")
        assert str(exc) == "custom message"

    def test_no_cache_found_error_has_default_message(self):
        assert "cache" in str(NoCacheFoundError()).lower()

    def test_no_vectorstore_found_error_has_default_message(self):
        assert "vectorstore" in str(NoVectorStoreFoundError()).lower()

    def test_vector_already_created_error_has_default_message(self):
        assert "already" in str(VectorAlreadyCreatedError()).lower()

    def test_all_are_subclasses_of_exception(self):
        for cls in (
            NoNewPDFError,
            NoCacheFoundError,
            NoVectorStoreFoundError,
            VectorAlreadyCreatedError,
        ):
            assert issubclass(cls, Exception)

    def test_exceptions_can_be_raised_and_caught(self):
        for cls in (
            NoNewPDFError,
            NoCacheFoundError,
            NoVectorStoreFoundError,
            VectorAlreadyCreatedError,
        ):
            with pytest.raises(cls):
                raise cls()
