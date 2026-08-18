import json
from pathlib import Path
import os
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag


# ============================================================
# APP SETTINGS
# ============================================================

STORY_TITLE = os.getenv(
    "STORY_TITLE",
    "The Adventure of the Speckled Band"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="StoryWorld API",
    description=(
        "Hybrid Story Memory + World Memory "
        "RAG backend."
    ),
    version="0.1.0"
)


# ============================================================
# CORS
#
# Next.js will normally run on port 3000.
# FastAPI will run on port 8000.
# ============================================================

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        (
            "http://localhost:3000,"
            "http://127.0.0.1:3000"
        )
    ).split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,

    allow_origins=
        frontend_origins,

    # Allows GitHub Codespaces forwarded
    # frontend URLs such as:
    #
    # https://<codespace>-3000.app.github.dev
    allow_origin_regex=
        r"https://.*\.app\.github\.dev",

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "Content-Type"
    ]
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    include_debug: bool = False


class StorySource(BaseModel):

    chunk_id: int | None

    chapter: str

    page_start: int | None

    page_end: int | None

    excerpt: str


class WorldEvidence(BaseModel):

    type: str

    text: str

    sources: list[dict[str, Any]]


class ChatResponse(BaseModel):

    answer: str

    sources: list[StorySource]

    world_evidence: (
        list[WorldEvidence]
        | None
    ) = None


class StoryInfo(BaseModel):

    title: str

    story_chunks: int

    world_items: int

    embedding_model: str

    generation_model: str

    status: str


class CharacterInfo(BaseModel):

    name: str

    description: str

    sources: list[dict[str, Any]]


# ============================================================
# GENERATION LOCK
#
# For the MVP we process one generation at a time.
# This keeps our shared local model/client simple and safe.
# ============================================================

generation_lock = Lock()


# ============================================================
# SOURCE HELPERS
# ============================================================

def clean_excerpt(
    text: str,
    maximum_length: int = 350
):

    cleaned = " ".join(
        text.split()
    )

    if (
        len(cleaned)
        <= maximum_length
    ):

        return cleaned


    return (
        cleaned[
            :maximum_length
        ].rstrip()
        + "..."
    )


# ============================================================
# STORY SOURCE BUILDER
# ============================================================

def build_story_sources(
    story_results
):

    sources = []

    seen_chunks = set()


    for result in story_results:

        chunk_id = result.get(
            "chunk_id"
        )


        # Avoid duplicate source cards.
        if chunk_id in seen_chunks:

            continue


        seen_chunks.add(
            chunk_id
        )


        sources.append(
            StorySource(

                chunk_id=
                    chunk_id,

                chapter=
                    result.get(
                        "chapter",
                        "Unknown"
                    ),

                page_start=
                    result.get(
                        "page_start"
                    ),

                page_end=
                    result.get(
                        "page_end"
                    ),

                excerpt=
                    clean_excerpt(
                        result.get(
                            "text",
                            ""
                        )
                    )
            )
        )


    return sources


# ============================================================
# WORLD DEBUG BUILDER
# ============================================================

def build_world_debug(
    world_results
):

    evidence = []


    for result in world_results:

        evidence.append(
            WorldEvidence(

                type=
                    result.get(
                        "type",
                        "unknown"
                    ),

                text=
                    result.get(
                        "text",
                        ""
                    ),

                sources=
                    result.get(
                        "sources",
                        []
                    )
            )
        )


    return evidence


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name":
            "StoryWorld API",

        "status":
            "running",

        "docs":
            "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health"
)
def health():

    return {
        "status":
            "ready",

        "story_chunks":
            len(
                rag.story_chunks
            ),

        "world_items":
            len(
                rag.world_items
            ),

        "embedding_model_loaded":
            True,

        "gemini_model":
            rag.LLM_MODEL
    }


# ============================================================
# STORY INFORMATION
# ============================================================

@app.get(
    "/api/story",
    response_model=StoryInfo
)
def story_info():

    return StoryInfo(

        title=
            STORY_TITLE,

        story_chunks=
            len(
                rag.story_chunks
            ),

        world_items=
            len(
                rag.world_items
            ),

        embedding_model=
            rag.EMBEDDING_MODEL,

        generation_model=
            rag.LLM_MODEL,

        status=
            "ready"
    )


# ============================================================
# CHARACTERS
# ============================================================

@app.get(
    "/api/characters",
    response_model=
        list[CharacterInfo]
)
def get_characters():

    characters = []


    for item in rag.world_items:

        if (
            item.get(
                "type"
            )
            != "character"
        ):

            continue


        name = item.get(
            "name",
            ""
        )


        # build_world_index.py normally stores
        # name separately, but this is a fallback.
        if not name:

            text = item.get(
                "text",
                ""
            )

            if text.startswith(
                "Character:"
            ):

                name = (
                    text
                    .split(
                        ".",
                        1
                    )[0]
                    .replace(
                        "Character:",
                        ""
                    )
                    .strip()
                )


        characters.append(
            CharacterInfo(

                name=
                    name,

                description=
                    item.get(
                        "description",
                        ""
                    ),

                sources=
                    item.get(
                        "sources",
                        []
                    )
            )
        )


    characters.sort(
        key=lambda character:
            character.name.lower()
    )


    return characters



# ============================================================
# RELATIONSHIPS
# ============================================================

@app.get("/api/relationships")
def get_relationships():

    world_file = Path(
        "data/world_semantic_resolved.json"
    )

    if not world_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Resolved World Memory not found."
        )

    with open(
        world_file,
        "r",
        encoding="utf-8"
    ) as file:

        world = json.load(file)

    relationships = []

    seen = set()

    for item in world.get(
        "relationships",
        []
    ):

        subject = item.get(
            "subject",
            ""
        ).strip()

        relation = item.get(
            "relation",
            ""
        ).strip()

        obj = item.get(
            "object",
            ""
        ).strip()

        explanation = item.get(
            "explanation",
            ""
        ).strip()

        if not subject or not obj:
            continue

        key = (
            subject.lower(),
            relation.lower(),
            obj.lower()
        )

        if key in seen:
            continue

        seen.add(key)

        relationships.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "explanation": explanation,
            "sources": item.get(
                "sources",
                []
            )
        })

    return relationships

# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest
):

    question = (
        request.question.strip()
    )


    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    try:

        # ----------------------------------------------------
        # Existing validated hybrid RAG pipeline
        # ----------------------------------------------------

        with generation_lock:

            (
                answer,
                story_results,
                world_results
            ) = rag.answer_question(
                question
            )


        # ----------------------------------------------------
        # Convert retrieved passages into frontend-friendly
        # source cards.
        # ----------------------------------------------------

        sources = (
            build_story_sources(
                story_results
            )
        )


        # ----------------------------------------------------
        # World Memory is hidden from normal users.
        #
        # include_debug=true can expose it while developing.
        # ----------------------------------------------------

        world_debug = None


        if request.include_debug:

            world_debug = (
                build_world_debug(
                    world_results
                )
            )


        return ChatResponse(

            answer=
                answer,

            sources=
                sources,

            world_evidence=
                world_debug
        )


    except Exception as error:

        # Log the real error in the server terminal.
        print()
        print(
            "StoryWorld generation error:"
        )

        print(
            repr(error)
        )

        print()


        # Avoid exposing internal exception contents
        # directly to the frontend.
        raise HTTPException(

            status_code=503,

            detail=(
                "StoryWorld could not generate "
                "an answer right now."
            )
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(
        "api:app",

        host=
            "0.0.0.0",

        port=
            8000,

        reload=
            True
    )