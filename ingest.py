from pathlib import Path
import json
import re

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer


BOOK_PATH = Path("books/sherlock.pdf")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pdf(path):
    print("Reading book...")

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text(
            "text",
            sort=True
        ).strip()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    print(f"Extracted {len(pages)} pages.")

    return pages


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def create_chunks(
    pages,
    chunk_size=180,
    overlap=40
):

    chunks = []
    chunk_id = 0

    for page in pages:

        text = clean_text(page["text"])

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            if len(chunk_words) >= 30:

                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(chunk_words)
                })

                chunk_id += 1

            start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_embeddings(chunks):

    print("Loading local embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


def save_data(chunks, embeddings):

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print("Saved story memory locally.")


def main():

    pages = extract_pdf(BOOK_PATH)

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    save_data(
        chunks,
        embeddings
    )

    print()
    print("StoryWorld memory created successfully.")


if __name__ == "__main__":
    main()