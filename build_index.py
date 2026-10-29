"""Build the search index from the documents folder.

Usage:
    python build_index.py                 # uses DOCS_DIR from .env (default: data/docs)
    python build_index.py --docs my_pdfs  # index a different folder
"""

import argparse
import time

from hallcheck.config import Settings
from hallcheck.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", help="folder with .txt, .md or .pdf files")
    args = parser.parse_args()

    settings = Settings()
    if args.docs:
        settings.docs_dir = args.docs
    start = time.perf_counter()
    n = Retriever(settings).build()
    print(f"Indexed {n} chunks from {settings.docs_dir} into {settings.index_dir} "
          f"in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
