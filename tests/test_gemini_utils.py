"""Tests for pure utility methods on the Gemini class (llms/gemini.py).

Gemini.__init__ calls GoogleGenerativeAIEmbeddings which requires real
credentials. These tests only exercise @staticmethod methods, so no
instantiation — and no credentials — are needed.
"""

import json

from llms.gemini import Gemini


class TestTestJsonValidity:
    def test_valid_object_returns_the_string(self):
        payload = json.dumps({"key": "value"})
        assert Gemini.test_json_validity(payload) == payload

    def test_valid_array_returns_the_string(self):
        payload = json.dumps([1, 2, 3])
        assert Gemini.test_json_validity(payload) == payload

    def test_nested_json_returns_the_string(self):
        payload = json.dumps({"a": {"b": [True, None, 42]}})
        assert Gemini.test_json_validity(payload) == payload

    def test_invalid_json_returns_none(self):
        assert Gemini.test_json_validity("not-json{{{") is None

    def test_empty_string_returns_none(self):
        assert Gemini.test_json_validity("") is None

    def test_bare_string_returns_none(self):
        # A bare Python string is not valid JSON
        assert Gemini.test_json_validity("hello") is None

    def test_valid_json_string_literal_returns_payload(self):
        # JSON-encoded string (with quotes) is valid JSON
        payload = json.dumps("just a string")
        assert Gemini.test_json_validity(payload) == payload
