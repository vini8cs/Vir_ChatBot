from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenizerWrapper(PreTrainedTokenizerBase):
    """
    Wrapper para tokenizers Hugging Face (ex: LLaMA, Mistral)
    compatível com o HybridChunker do docling
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", max_length: int = 2048, **kwargs):
        super().__init__(model_max_length=max_length, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self._vocab_size = self.tokenizer.vocab_size
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model_name = args[0]
        return cls(model_name=model_name, **kwargs)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, **kwargs)
        return [str(t) for t in token_ids]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self):
        return self.vocab_size

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        return self.tokenizer.save_vocabulary(save_directory, **kwargs)
