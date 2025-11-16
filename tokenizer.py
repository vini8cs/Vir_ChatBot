from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from transformers import AutoTokenizer


class TokenizerWrapper(BaseTokenizer):
    """
    Hugging Face tokenizer wrapper compatible with Docling HybridChunker.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", max_length=8191):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.model_max_length = max_length
        self._max_length = max_length
        self._model_name = model_name

    def get_tokenizer(self):
        """Return the underlying tokenizer instance."""
        return self._tokenizer

    def get_max_tokens(self) -> int:
        """Return the maximum token length."""
        return self._max_length

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return self._tokenizer.decode(ids)

    def count_tokens(self, text: str) -> int:
        """Return token count for a given text."""
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
