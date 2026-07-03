"""
Builds the ChromaDB policy index from data/policies/*.md.

Run with: python scripts/build_policy_index.py

Each policy doc is small and self-contained (per CLAUDE.md's "keep the corpus
small and clean" guidance) so it's embedded as a single chunk - no further
splitting. Persisted to data/policy_index/ so agents/policy_assessor.py can
load it without re-embedding on every run.

Requires OPENAI_API_KEY in the environment (used for embeddings).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

import chromadb
from chromadb.utils import embedding_functions

POLICIES_DIR = ROOT / "data" / "policies"
INDEX_PATH = ROOT / "data" / "policy_index"
COLLECTION_NAME = "finops_policies"
EMBEDDING_MODEL = "text-embedding-3-small"


def build_index(policies_dir: Path = POLICIES_DIR, index_path: Path = INDEX_PATH) -> None:
    docs = sorted(policies_dir.glob("*.md"))
    if not docs:
        raise RuntimeError(f"No policy docs found in {policies_dir}")

    if index_path.exists():
        import shutil

        shutil.rmtree(index_path)
    index_path.mkdir(parents=True)

    client = chromadb.PersistentClient(path=str(index_path))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name=EMBEDDING_MODEL
    )
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    collection.add(
        ids=[doc.stem for doc in docs],
        documents=[doc.read_text() for doc in docs],
        metadatas=[{"filename": doc.name} for doc in docs],
    )

    print(f"Indexed {len(docs)} policy docs into {index_path} (collection={COLLECTION_NAME!r}).")


if __name__ == "__main__":
    build_index()
