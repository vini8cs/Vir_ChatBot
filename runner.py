from agents.vir_chatbot.vectorstore import VectorStoreCreator


def _create_vs(args):
    """Internal helper to build the VectorStoreCreator with given args."""
    return VectorStoreCreator(
        vectorstore_path=args["vectorstore_path"],
        pdf_folder=args["pdf_folder"],
        output_folder=args["temp_folder"],
        cache=args["cache"],
        gemini_model=args["gemini_model"],
        embedding_model=args["embedding_model"],
        temperature=args["temperature"],
        max_output_tokens=args["max_output_tokens"],
        token_size=args["max_tokens_for_chunk"],
        languages=args["languages"],
        pdfs_to_delete=args["pdfs_to_delete"],
        tokenizer_model=args["tokenizer_model"],
        threads=args["threads"],
        dont_summarize=args["dont_summarize"],
    )


def run_build(args, callback=None):
    """
    Build a brand-new vectorstore from a folder of PDFs.
    """
    vs = _create_vs(args)
    if callback:
        vs.set_callback(callback)
    vs.build_vectorstore_from_folder()


def run_add(args, callback=None):
    """
    Add new PDFs to an existing vectorstore.
    """
    vs = _create_vs(args)
    if callback:
        vs.set_callback(callback)
    vs.add_from_folder()


def run_delete(args):
    """
    Delete selected PDFs from the vectorstore.
    """
    vs = _create_vs(args)
    vs.delete_pdfs()
