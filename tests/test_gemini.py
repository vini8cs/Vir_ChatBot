"""Tests for Gemini instance methods in llms/gemini.py.

``langchain_google_genai`` is stubbed in conftest.py, so construction
produces MagicMocks for ChatGoogleGenerativeAI / Embeddings — we assert
against that instead of against a real SDK.
"""

import json
from unittest.mock import MagicMock

from llms.gemini import Gemini


def _make_gemini(prompt_text=None, prompt_image=None):
    return Gemini(
        gemini_model="gemini-2.5-flash",
        temperature=0.1,
        max_output_tokens=512,
        response_schema={"type": "object"},
        max_retries=2,
        prompt_text=prompt_text,
        prompt_image=prompt_image,
    )


class TestInit:
    def test_init_without_prompts_leaves_chains_none(self):
        g = _make_gemini()
        assert g.summarize_chain_text is None
        assert g.summarize_chain_image is None
        assert g.prompt is None

    def test_init_with_prompt_text_creates_text_chain(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        assert g.summarize_chain_text is not None
        assert g.prompt is not None

    def test_init_with_prompt_image_creates_image_chain(self):
        g = _make_gemini(prompt_image="Describe image")
        assert g.summarize_chain_image is not None

    def test_stores_configured_embedding_model(self):
        g = Gemini(gemini_embedding_model="custom-embed-model")
        assert g.gemini_embedding_model == "custom-embed-model"


class TestGenerateTextSummaries:
    def test_returns_none_for_empty_content(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        assert g._generate_text_summaries("") is None

    def test_returns_none_when_content_has_few_words(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        # clean_text requires > 3 words — exactly 3 returns None.
        assert g._generate_text_summaries("only three words") is None

    def test_returns_response_when_json_valid(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        payload = json.dumps([{"summary": "s", "isReference": False}])
        response = MagicMock()
        response.content = payload
        g.summarize_chain_text = MagicMock()
        g.summarize_chain_text.invoke.return_value = response

        result = g._generate_text_summaries(
            "one two three four five words in content"
        )
        assert result == payload

    def test_returns_fallback_when_json_invalid(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        response = MagicMock()
        response.content = "not json at all"
        g.summarize_chain_text = MagicMock()
        g.summarize_chain_text.invoke.return_value = response

        result = g._generate_text_summaries(
            "one two three four five six seven"
        )
        assert isinstance(result, str)
        assert json.loads(result) == {
            "summary": "one two three four five six seven",
            "isReference": False,
        }

    def test_strips_bullet_unicode_from_content(self):
        g = _make_gemini(prompt_text="Summarize: {element}")
        response = MagicMock()
        response.content = json.dumps(
            [{"summary": "x", "isReference": False}]
        )
        g.summarize_chain_text = MagicMock()
        g.summarize_chain_text.invoke.return_value = response

        g._generate_text_summaries("• one • two three four five")
        called_with = g.summarize_chain_text.invoke.call_args[0][0]
        assert "•" not in called_with["element"]


class TestGenerateImageSummaries:
    def test_returns_none_when_response_empty(self):
        g = _make_gemini(prompt_image="Describe")
        response = MagicMock()
        response.content = ""
        g.summarize_chain_image = MagicMock()
        g.summarize_chain_image.invoke.return_value = response

        assert g._genenate_image_summaries("base64data") is None

    def test_returns_none_when_response_whitespace(self):
        g = _make_gemini(prompt_image="Describe")
        response = MagicMock()
        response.content = "   "
        g.summarize_chain_image = MagicMock()
        g.summarize_chain_image.invoke.return_value = response

        assert g._genenate_image_summaries("base64data") is None

    def test_returns_payload_when_valid_json(self):
        g = _make_gemini(prompt_image="Describe")
        payload = json.dumps([{"summary": "desc", "isReference": False}])
        response = MagicMock()
        response.content = payload
        g.summarize_chain_image = MagicMock()
        g.summarize_chain_image.invoke.return_value = response

        assert g._genenate_image_summaries("base64data") == payload

    def test_returns_none_on_invalid_json(self):
        g = _make_gemini(prompt_image="Describe")
        response = MagicMock()
        response.content = "not-json{"
        g.summarize_chain_image = MagicMock()
        g.summarize_chain_image.invoke.return_value = response

        assert g._genenate_image_summaries("base64data") is None

    def test_builds_human_message_with_image_and_text(self, mocker):
        g = _make_gemini(prompt_image="Describe the image")
        response = MagicMock()
        response.content = json.dumps(
            [{"summary": "x", "isReference": False}]
        )
        g.summarize_chain_image = MagicMock()
        g.summarize_chain_image.invoke.return_value = response

        g._genenate_image_summaries("base64data")
        args, _ = g.summarize_chain_image.invoke.call_args
        message_list = args[0]
        assert len(message_list) == 1
        msg_content = message_list[0].content
        assert any(
            part.get("type") == "image_url"
            and "base64data" in part["image_url"]["url"]
            for part in msg_content
        )
        assert any(
            part.get("type") == "text"
            and part.get("text") == "Describe the image"
            for part in msg_content
        )


class TestCreateSummarizedChains:
    """Cover the code paths that instantiate ChatGoogleGenerativeAI."""

    def test_text_chain_is_built_when_prompt_text_provided(self):
        g = _make_gemini(prompt_text="tmpl: {element}")
        assert g.summarize_chain_text is not None

    def test_image_chain_is_built_when_prompt_image_provided(self):
        g = _make_gemini(prompt_image="describe")
        assert g.summarize_chain_image is not None
