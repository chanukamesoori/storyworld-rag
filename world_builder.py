import json
import time
import re
from pathlib import Path

from google import genai
from pydantic import BaseModel
from typing import List


# ============================================================
# SETTINGS
# ============================================================

CHUNKS_FILE = Path(
    "data/chunks.json"
)

WORLD_FILE = Path(
    "data/world.json"
)

PROGRESS_FILE = Path(
    "data/world_progress.json"
)

LLM_MODEL = "gemini-3.6-flash"


# ------------------------------------------------------------
# FIRST TEST ONLY
#
# Keep this at 20 until we inspect the World Memory.
#
# Later:
#
# MAX_CHUNKS = None
# ------------------------------------------------------------

MAX_CHUNKS = 20


# Delay between successful requests
REQUEST_DELAY_SECONDS = 4


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class CharacterExtraction(BaseModel):

    name: str

    description: str


class LocationExtraction(BaseModel):

    name: str

    description: str


class RelationshipExtraction(BaseModel):

    subject: str

    relation: str

    object: str

    explanation: str


class EventExtraction(BaseModel):

    summary: str

    participants: List[str]

    location: str

    explanation: str


class FactExtraction(BaseModel):

    subject: str

    predicate: str

    object: str

    explanation: str


class WorldExtraction(BaseModel):

    characters: List[
        CharacterExtraction
    ]

    locations: List[
        LocationExtraction
    ]

    relationships: List[
        RelationshipExtraction
    ]

    events: List[
        EventExtraction
    ]

    facts: List[
        FactExtraction
    ]


# ============================================================
# GEMINI
# ============================================================

client = genai.Client()


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"{CHUNKS_FILE} does not exist. "
            "Run ingest.py first."
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
        )

    print(
        f"Loaded {len(chunks)} "
        f"story chunks."
    )

    return chunks


# ============================================================
# EMPTY WORLD
# ============================================================

def create_empty_world():

    return {

        "processed_chunk_ids": [],

        "characters": [],

        "locations": [],

        "relationships": [],

        "events": [],

        "facts": []
    }


# ============================================================
# LOAD PROGRESS
# ============================================================

def load_world():

    if PROGRESS_FILE.exists():

        print(
            "Existing World Memory "
            "progress found."
        )

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    return create_empty_world()


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(world):

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            world,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# SAVE FINAL
# ============================================================

def save_final_world(world):

    with open(
        WORLD_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            world,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    return " ".join(
        text.lower()
        .strip()
        .split()
    )


# ============================================================
# SOURCE METADATA
# ============================================================

def get_source(chunk):

    return {

        "chunk_id":
            chunk.get(
                "chunk_id"
            ),

        "chapter":
            chunk.get(
                "chapter",
                "Unknown"
            ),

        "page_start":
            chunk.get(
                "page_start",
                chunk.get("page")
            ),

        "page_end":
            chunk.get(
                "page_end",
                chunk.get("page")
            )
    }


# ============================================================
# CHARACTER
# ============================================================

def add_character(
    world,
    character,
    source
):

    name_key = normalize(
        character.name
    )

    if not name_key:
        return

    for existing in (
        world["characters"]
    ):

        if normalize(
            existing["name"]
        ) == name_key:

            if (
                source
                not in existing["sources"]
            ):

                existing[
                    "sources"
                ].append(
                    source
                )

            return

    world[
        "characters"
    ].append({

        "name":
            character.name,

        "description":
            character.description,

        "sources": [
            source
        ]
    })


# ============================================================
# LOCATION
# ============================================================

def add_location(
    world,
    location,
    source
):

    name_key = normalize(
        location.name
    )

    if not name_key:
        return

    for existing in (
        world["locations"]
    ):

        if normalize(
            existing["name"]
        ) == name_key:

            if (
                source
                not in existing["sources"]
            ):

                existing[
                    "sources"
                ].append(
                    source
                )

            return

    world[
        "locations"
    ].append({

        "name":
            location.name,

        "description":
            location.description,

        "sources": [
            source
        ]
    })


# ============================================================
# RELATIONSHIP
# ============================================================

def add_relationship(
    world,
    relationship,
    source
):

    key = (

        normalize(
            relationship.subject
        ),

        normalize(
            relationship.relation
        ),

        normalize(
            relationship.object
        )
    )

    if not all(key):
        return

    for existing in (
        world["relationships"]
    ):

        existing_key = (

            normalize(
                existing["subject"]
            ),

            normalize(
                existing["relation"]
            ),

            normalize(
                existing["object"]
            )
        )

        if existing_key == key:

            if (
                source
                not in existing["sources"]
            ):

                existing[
                    "sources"
                ].append(
                    source
                )

            return

    world[
        "relationships"
    ].append({

        "subject":
            relationship.subject,

        "relation":
            relationship.relation,

        "object":
            relationship.object,

        "explanation":
            relationship.explanation,

        "sources": [
            source
        ]
    })


# ============================================================
# EVENT
# ============================================================

def add_event(
    world,
    event,
    source
):

    key = normalize(
        event.summary
    )

    if not key:
        return

    for existing in (
        world["events"]
    ):

        if normalize(
            existing["summary"]
        ) == key:

            if (
                source
                not in existing["sources"]
            ):

                existing[
                    "sources"
                ].append(
                    source
                )

            return

    world[
        "events"
    ].append({

        "summary":
            event.summary,

        "participants":
            event.participants,

        "location":
            event.location,

        "explanation":
            event.explanation,

        "sources": [
            source
        ]
    })


# ============================================================
# FACT
# ============================================================

def add_fact(
    world,
    fact,
    source
):

    key = (

        normalize(
            fact.subject
        ),

        normalize(
            fact.predicate
        ),

        normalize(
            fact.object
        )
    )

    if not all(key):
        return

    for existing in (
        world["facts"]
    ):

        existing_key = (

            normalize(
                existing["subject"]
            ),

            normalize(
                existing["predicate"]
            ),

            normalize(
                existing["object"]
            )
        )

        if existing_key == key:

            if (
                source
                not in existing["sources"]
            ):

                existing[
                    "sources"
                ].append(
                    source
                )

            return

    world[
        "facts"
    ].append({

        "subject":
            fact.subject,

        "predicate":
            fact.predicate,

        "object":
            fact.object,

        "explanation":
            fact.explanation,

        "sources": [
            source
        ]
    })


# ============================================================
# MERGE EXTRACTION
# ============================================================

def merge_extraction(
    world,
    extraction,
    chunk
):

    source = get_source(
        chunk
    )

    for character in (
        extraction.characters
    ):

        add_character(
            world,
            character,
            source
        )

    for location in (
        extraction.locations
    ):

        add_location(
            world,
            location,
            source
        )

    for relationship in (
        extraction.relationships
    ):

        add_relationship(
            world,
            relationship,
            source
        )

    for event in (
        extraction.events
    ):

        add_event(
            world,
            event,
            source
        )

    for fact in (
        extraction.facts
    ):

        add_fact(
            world,
            fact,
            source
        )


# ============================================================
# EXTRACT ONE CHUNK
# ============================================================

def extract_world_from_chunk(
    chunk
):

    chapter = chunk.get(
        "chapter",
        "Unknown"
    )

    page_start = chunk.get(
        "page_start",
        chunk.get("page")
    )

    page_end = chunk.get(
        "page_end",
        chunk.get("page")
    )

    story_text = (
        chunk["text"]
    )

    prompt = f"""
You are building a structured database representing
a fictional story world.

You are NOT answering a user question.

Extract only information supported by the supplied
story passage.

RULES:

1. Use ONLY the supplied passage.

2. Do NOT use prior knowledge of the book.

3. Do NOT invent information.

4. If something is uncertain, leave it out.

5. Extract characters actually mentioned or present.

6. Extract meaningful locations.

7. Extract meaningful relationships between entities.

8. Extract important events.

9. Extract useful factual statements.

10. Use short machine-readable relationship names.

Examples:

friend_of
companion_of
lives_at
works_with
knows
married_to
parent_of
enemy_of
respects
suspects
owns
visited
helped
speaks_to
travels_with

11. Use short machine-readable fact predicates.

Examples:

occupation
residence
appearance
possesses
believes
knows
status
identity

12. The explanation must briefly state what evidence
supports the extraction.

13. Do not create facts merely because they are commonly
known about the story.

14. Return empty arrays where no useful information exists.

STORY METADATA

Chapter:
{chapter}

Pages:
{page_start} - {page_end}

STORY PASSAGE:

{story_text}
"""

    interaction = (
        client.interactions.create(

            model=LLM_MODEL,

            input=prompt,

            response_format={

                "type":
                    "text",

                "mime_type":
                    "application/json",

                "schema":
                    WorldExtraction
                    .model_json_schema()
            },

            store=False
        )
    )

    extraction = (
        WorldExtraction
        .model_validate_json(
            interaction.output_text
        )
    )

    return extraction


# ============================================================
# EXTRACT RETRY TIME FROM GEMINI ERROR
# ============================================================

def get_retry_seconds(
    error_message
):

    # Example:
    # "Please retry in 45.329404514s"

    match = re.search(
        r"retry in ([0-9.]+)s",
        error_message,
        re.IGNORECASE
    )

    if match:

        seconds = float(
            match.group(1)
        )

        # Add a small safety margin
        return seconds + 2

    # Default fallback
    return 60


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(world):

    print()
    print(
        "Current World Memory:"
    )

    print(
        f"Characters    : "
        f"{len(world['characters'])}"
    )

    print(
        f"Locations     : "
        f"{len(world['locations'])}"
    )

    print(
        f"Relationships : "
        f"{len(world['relationships'])}"
    )

    print(
        f"Events        : "
        f"{len(world['events'])}"
    )

    print(
        f"Facts         : "
        f"{len(world['facts'])}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "STORYWORLD — WORLD MEMORY BUILDER"
    )

    print("=" * 70)
    print()

    chunks = load_chunks()

    world = load_world()

    processed_ids = set(
        world[
            "processed_chunk_ids"
        ]
    )

    # --------------------------------------------------------
    # TEST LIMIT
    # --------------------------------------------------------

    if MAX_CHUNKS is not None:

        chunks_to_process = (
            chunks[:MAX_CHUNKS]
        )

    else:

        chunks_to_process = (
            chunks
        )

    total = len(
        chunks_to_process
    )

    print(
        f"Chunks selected for this run: "
        f"{total}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS CHUNKS
    # --------------------------------------------------------

    for position, chunk in enumerate(
        chunks_to_process,
        start=1
    ):

        chunk_id = (
            chunk["chunk_id"]
        )

        # ----------------------------------------------------
        # Already completed
        # ----------------------------------------------------

        if chunk_id in processed_ids:

            print(
                f"[{position}/{total}] "
                f"Chunk {chunk_id} "
                f"already processed."
            )

            continue

        print()
        print(
            "-" * 70
        )

        print(
            f"[{position}/{total}] "
            f"Processing chunk "
            f"{chunk_id}"
        )

        print(
            f"Chapter: "
            f"{chunk.get('chapter', 'Unknown')}"
        )

        print(
            f"Pages: "
            f"{chunk.get('page_start', chunk.get('page'))}"
            f" → "
            f"{chunk.get('page_end', chunk.get('page'))}"
        )

        extraction = None

        # ----------------------------------------------------
        # RETRIES
        # ----------------------------------------------------

        for attempt in range(
            1,
            4
        ):

            try:

                extraction = (
                    extract_world_from_chunk(
                        chunk
                    )
                )

                break

            except Exception as error:

                error_text = str(
                    error
                )

                print(
                    f"Attempt {attempt} failed:"
                )

                print(
                    error_text
                )

                # ------------------------------------------------
                # Gemini rate limit
                # ------------------------------------------------

                if "429" in error_text:

                    retry_seconds = (
                        get_retry_seconds(
                            error_text
                        )
                    )

                    if attempt < 3:

                        print(
                            f"Rate limit reached."
                        )

                        print(
                            f"Retrying after "
                            f"{retry_seconds:.0f} "
                            f"seconds..."
                        )

                        time.sleep(
                            retry_seconds
                        )

                elif attempt < 3:

                    print(
                        "Retrying after 5 seconds..."
                    )

                    time.sleep(
                        5
                    )

        # ----------------------------------------------------
        # FAILED COMPLETELY
        # ----------------------------------------------------

        if extraction is None:

            print()
            print(
                "Could not process this chunk."
            )

            print(
                "All completed progress "
                "has already been saved."
            )

            save_progress(
                world
            )

            save_final_world(
                world
            )

            break

        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        merge_extraction(
            world,
            extraction,
            chunk
        )

        # ----------------------------------------------------
        # MARK COMPLETE
        # ----------------------------------------------------

        world[
            "processed_chunk_ids"
        ].append(
            chunk_id
        )

        processed_ids.add(
            chunk_id
        )

        # ----------------------------------------------------
        # SAVE AFTER EVERY CHUNK
        # ----------------------------------------------------

        save_progress(
            world
        )

        print(
            "✓ World information extracted"
        )

        print_statistics(
            world
        )

        # ----------------------------------------------------
        # SMALL DELAY BETWEEN REQUESTS
        # ----------------------------------------------------

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    save_final_world(
        world
    )

    print()
    print("=" * 70)

    print(
        "WORLD MEMORY BUILD COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Saved to: "
        f"{WORLD_FILE}"
    )

    print_statistics(
        world
    )


if __name__ == "__main__":

    main()