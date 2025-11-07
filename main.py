import argparse

from vectorstore import VectorStoreCreator

# import config


def menu():
    parser = argparse.ArgumentParser(description="VirChatBot Command Line Interface")
    parser.add_argument("--vectorstore_path", type=str, default="vectorstore", help="Path to the vectorstore folder")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Path to the folder containing PDF files")
    parser.add_argument("--temp_folder", type=str, default="temp_folder", help="Path to the temporary folder")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--embedding_model", type=str, default="gemini-embedding-001", help="Embedding model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for the ai model")
    parser.add_argument("--max_output_tokens", type=int, default=2048, help="Max output tokens for the ai model")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Chunk size for chunking PDFs with Unstructured")
    parser.add_argument("--cache", type=str, default="vectorstore_cache.csv", help="Path to the cache CSV")
    parser.add_argument(
        "--combine_text_under_n_chars",
        type=int,
        default=1500,
        help="Combine text elements under this number of characters",
    )
    parser.add_argument(
        "--new_after_n_characters", type=int, default=3000, help="Start a new chunk after this number of characters"
    )
    parser.add_argument("--languages", type=str, nargs="+", default=["eng", "pt"], help="Languages for OCR processing")
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
        chunk_size=args.chunk_size,
        combine_text_under_n_chars=args.combine_text_under_n_chars,
        new_after_n_characters=args.new_after_n_characters,
        languages=args.languages,
    )

    vectorstore.create_vectorstore()


if __name__ == "__main__":
    main()
