import argparse

from agents.vir_chatbot.vectorstore import VectorStoreCreator

# import config


def menu():
    parser = argparse.ArgumentParser(description="VirChatBot Command Line Interface")

    parser.add_argument(
        "--build_vectorstore",
        action="store_true",
        help="Rebuild the vectorstore from all PDFs in the folder and overwrite cache",
    )

    parser.add_argument(
        "--add_pdfs",
        action="store_true",
        help="Add new PDFs (not present in cache) to the existing vectorstore",
    )

    parser.add_argument("--vectorstore_path", type=str, default="vectorstore", help="Path to the vectorstore folder")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Path to the folder containing PDF files")
    parser.add_argument("--temp_folder", type=str, default="temp_folder", help="Path to the temporary folder")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--embedding_model", type=str, default="gemini-embedding-001", help="Embedding model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for the ai model")
    parser.add_argument("--max_output_tokens", type=int, default=2048, help="Max output tokens for the ai model")
    parser.add_argument(
        "--max_tokens_for_chunk", type=int, default=2048, help="Token size for chunking PDFs with the tokenizer"
    )
    parser.add_argument("--cache", type=str, default="vectorstore_cache.csv", help="Path to the cache CSV")
    parser.add_argument("--languages", type=str, nargs="+", default=["eng", "pt"], help="Languages for OCR processing")
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Tokenizer model to use. It should be a Hugging Face model from transformers library.",
    )
    parser.add_argument(
        "--pdfs_to_delete",
        type=str,
        nargs="+",
        default=[],
        help="List of PDF filenames (e.g. /path/to/file /path/to/file2) to delete from the vectorstore and cache.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for PDF processing.",
    )
    parser.add_argument(
        "--dont_summarize",
        action="store_true",
        help="If set, the PDFs will be added to the vectorstore without summarization.",
    )
    args = parser.parse_args()
    return args


def main():
    args = menu()
    vectorstore = VectorStoreCreator(
        vectorstore_path=args.vectorstore_path,
        pdf_folder=args.pdf_folder,
        output_folder=args.temp_folder,
        cache=args.cache,
        gemini_model=args.gemini_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        token_size=args.max_tokens_for_chunk,
        languages=args.languages,
        pdfs_to_delete=args.pdfs_to_delete,
        tokenizer_model=args.tokenizer_model,
        threads=args.threads,
        dont_summarize=args.dont_summarize,
    )

    if args.build_vectorstore:
        vectorstore.build_vectorstore_from_folder()
    elif args.add_pdfs:
        vectorstore.add_from_folder()
    elif args.pdfs_to_delete:
        vectorstore.delete_pdfs()


if __name__ == "__main__":
    main()
