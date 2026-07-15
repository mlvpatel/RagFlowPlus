"""Load the bundled sample documents into rag-advanced-2023.

For a fully local, keyless run with Ollama:

    ollama serve &
    ollama pull nomic-embed-text
    EMBEDDING_PROVIDER=ollama python scripts/load_sample_data.py

Each file is chunked, embedded with the configured provider, and stored in
Chroma, exactly as an upload through the UI would be.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.db_utils import insert_document_record  # noqa: E402
from src.embeddings.chroma_utils import index_document_to_chroma  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def main() -> None:
    files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt sample files found in {SAMPLE_DIR}")
        return
    print(f"Loading {len(files)} sample documents from {SAMPLE_DIR.name}/")
    for path in files:
        file_id = insert_document_record(path.name)
        ok = index_document_to_chroma(str(path), file_id)
        print(f"  {path.name}: {'indexed' if ok else 'failed'} (file_id={file_id})")
    print("Done. Start the UI and ask a question, see sample_data/README.md.")


if __name__ == "__main__":
    main()
