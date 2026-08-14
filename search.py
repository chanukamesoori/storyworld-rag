import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

CHUNKS_FILE = "data/chunks.json"
EMBEDDINGS_FILE = "data/embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD STORY MEMORY
# ============================================================

print("Loading StoryWorld memory...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


embeddings = np.load(
    EMBEDDINGS_FILE
)

print(f"Loaded {len(chunks)} story chunks.")


# ============================================================
# LOAD SAME EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_story(
    question,
    top_k=5
):

    # Convert user's question into an embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )


    # Compare the question against EVERY story chunk
    scores = np.dot(
        embeddings,
        question_embedding
    )


    # Get the indexes of the highest scores
    best_indexes = np.argsort(
        scores
    )[::-1][:top_k]


    results = []

    for index in best_indexes:

        results.append({
            "score": float(scores[index]),
            "page": chunks[index]["page"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"]
        })


    return results


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    print()
    print("=" * 70)

    question = input(
        "Ask something about the story (or type 'exit'): "
    )

    if question.lower() == "exit":
        break


    results = search_story(
        question,
        top_k=5
    )


    print()
    print("Most relevant story passages:")
    print()


    for number, result in enumerate(
        results,
        start=1
    ):

        print("-" * 70)

        print(
            f"RESULT {number}"
        )

        print(
            f"Similarity score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print()

        print(
            result["text"]
        )

        print()