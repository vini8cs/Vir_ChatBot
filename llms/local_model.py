import logging

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

import config as _

logger = logging.getLogger(__name__)

# Singleton caches — each model loads once per process and stays in memory.
# Re-loading a 7B LLM costs 30-60s and re-fills VRAM.
# Re-loading an embedding model costs ~5s and uses ~1GB VRAM.
_cached_llm: ChatHuggingFace | None = None
_cached_model_name: str | None = None

_cached_embeddings: HuggingFaceEmbeddings | None = None
_cached_embedding_model: str | None = None


def get_local_llm(model_name: str, bits: int = 4) -> ChatHuggingFace:
    """
    Load a quantized local LLM and return a LangChain-compatible chat model.

    Uses bitsandbytes NF4 quantization (4-bit Normal Float):
    - 4-bit: Qwen2.5-7B uses ~4.3 GB VRAM (fits in GTX 1060 6 GB)
    - double quantization saves an extra ~0.4 bits per weight
    - NF4 data type is optimal for normally distributed LLM weights

    The model is cached as a module-level singleton: subsequent calls with
    the same model_name return the already-loaded instance immediately.
    A different model_name triggers a full reload (previous model is evicted
    from VRAM automatically via Python GC + CUDA cache clear).
    """
    global _cached_llm, _cached_model_name

    if _cached_llm is not None and _cached_model_name == model_name:
        logger.info("Returning cached local model: %s", model_name)
        return _cached_llm

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. A NVIDIA GPU is required for local model "
            "inference. Set LLM_PROVIDER=gemini in your .env to use the "
            "Gemini API instead."
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
    logger.info(
        "Loading '%s' with %d-bit quantization on %s (%d GB VRAM)...",
        model_name,
        bits,
        device_name,
        total_vram,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        # Use float16 for matrix multiplications (dequantized compute).
        # The GTX 1060 does not have bfloat16 hardware support (Pascal arch),
        # so float16 is the correct choice here.
        bnb_4bit_compute_dtype=torch.float16,
        # Double quantization: quantize the quantization constants themselves,
        # saving ~0.37 extra bits per parameter with negligible quality loss.
        bnb_4bit_use_double_quant=True,
        # NF4 (Normal Float 4): designed for weights that follow a normal
        # distribution, which LLM weights do. Outperforms plain INT4.
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=_.LOCAL_MODELS_PATH,
    )

    # If reloading a different model, free the previous one first.
    if _cached_llm is not None:
        logger.info(
            "Evicting previous model from VRAM before loading new one."
        )
        global _cached_model_name  # noqa: PLW0621
        _cached_llm = None
        _cached_model_name = None
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda",
        trust_remote_code=True,
        cache_dir=_.LOCAL_MODELS_PATH,
    )

    # streaming=True uses TextIteratorStreamer under the hood, which allows
    # LangGraph's astream_events to emit on_chat_model_stream events
    # token-by-token — the same streaming behaviour as the Gemini provider.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=_.TEMPERATURE,
        do_sample=True,
        # Return only the newly generated tokens, not the prompt.
        return_full_text=False,
    )

    hf_pipeline = HuggingFacePipeline(pipeline=pipe, streaming=True)

    # ChatHuggingFace wraps HuggingFacePipeline with:
    # - Proper chat template formatting (Qwen2.5 has an excellent one)
    # - bind_tools() support via the chat template's tool-call format
    # - Standard AIMessage / ToolMessage return types expected by LangGraph
    chat_model = ChatHuggingFace(llm=hf_pipeline, verbose=False)

    _cached_llm = chat_model
    _cached_model_name = model_name

    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(
        "Model '%s' ready. VRAM: %.1f / %.1f GB used.",
        model_name,
        vram_used,
        vram_total,
    )

    return chat_model


def get_local_embeddings(model_name: str) -> Embeddings:
    """
    Load a local sentence-transformers embedding model and return a
    LangChain-compatible Embeddings object.

    Uses BAAI/bge-m3 by default (multilingual, ~1.1 GB VRAM in FP16).
    Cached as singleton — loads once per process.

    VRAM budget on GTX 1060 6 GB:
      Qwen2.5-7B (4-bit):  ~4.3 GB
      bge-m3 (FP16):       ~1.1 GB
      Total:               ~5.4 GB  ← fits within 6 GB
    """
    global _cached_embeddings, _cached_embedding_model

    if (
        _cached_embeddings is not None
        and _cached_embedding_model == model_name
    ):
        logger.info("Returning cached local embeddings: %s", model_name)
        return _cached_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading local embedding model '%s' on %s...", model_name, device
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=_.LOCAL_MODELS_PATH,
    )

    _cached_embeddings = embeddings
    _cached_embedding_model = model_name
    logger.info("Local embedding model '%s' ready.", model_name)

    return embeddings
