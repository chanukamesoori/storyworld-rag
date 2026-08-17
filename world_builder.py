import json
import re
import time
from pathlib import Path
from typing import List

from google import genai
from pydantic import BaseModel


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


# ============================================================
# FULL BOOK
#
# None = process every remaining chunk
#
# You can temporarily use:
# MAX_CHUNKS = 50
#
# if you ever want to limit one run.
# ============================================================

MAX_CHUNKS = None


# Paid API means we do not need aggressive free-tier
# optimization, but a small delay is still polite and
# reduces the chance of RPM spikes.
REQUEST_DELAY_SECONDS = 1

MAX_RETRIES = 3


# ============================================================
# STRUCTURED OUTPUT
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

    characters: List[CharacterExtraction]

    locations: List[LocationExtraction]

    relationships: List[RelationshipExtraction]

    events: List[EventExtraction]

    facts: List[FactExtraction]


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


# ============================================================
# EMPTY WORLD
# ============================================================

def create_empty_world():

    return {

        "processed_chunk_ids":
            [],

        "characters":
            [],

        "locations":
            [],

        "relationships":
            [],

        "events":
            [],

        "facts":
            []
    }


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. "
            "Run ingest.py first."
        )


    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

def load_world():

    # --------------------------------------------------------
    # Prefer checkpoint because it represents the latest
    # successfully processed state.
    # --------------------------------------------------------

    if PROGRESS_FILE.exists():

        print(
            f"Loading progress from "
            f"{PROGRESS_FILE}"
        )

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )


    # --------------------------------------------------------
    # Fallback to existing world
    # --------------------------------------------------------

    if WORLD_FILE.exists():

        print(
            f"No progress file found."
        )

        print(
            f"Loading existing world from "
            f"{WORLD_FILE}"
        )

        with open(
            WORLD_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )


    print(
        "No previous World Memory found."
    )

    print(
        "Starting from scratch."
    )

    return create_empty_world()


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_progress(
    world
):

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
# SAVE FINAL WORLD
# ============================================================

def save_world(
    world
):

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
# SOURCE METADATA
# ============================================================

def make_source(
    chunk
):

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
                chunk.get(
                    "page"
                )
            ),

        "page_end":
            chunk.get(
                "page_end",
                chunk.get(
                    "page"
                )
            )
    }


# ============================================================
# MERGE SOURCE LISTS
# ============================================================

def merge_sources(
    existing_sources,
    incoming_sources
):

    existing_ids = {

        source.get(
            "chunk_id"
        )

        for source
        in existing_sources
    }


    for source in incoming_sources:

        chunk_id = source.get(
            "chunk_id"
        )


        if chunk_id not in existing_ids:

            existing_sources.append(
                source
            )

            existing_ids.add(
                chunk_id
            )


# ============================================================
# CHARACTER MERGE
# ============================================================

def add_character(
    world,
    character,
    source
):

    key = normalize(
        character.name
    )


    if not key:
        return


    for existing in world[
        "characters"
    ]:

        if normalize(
            existing.get(
                "name"
            )
        ) == key:

            merge_sources(
                existing.setdefault(
                    "sources",
                    []
                ),
                [source]
            )


            # Keep the more informative description.
            if (
                len(
                    character.description
                )
                >
                len(
                    existing.get(
                        "description",
                        ""
                    )
                )
            ):

                existing[
                    "description"
                ] = character.description


            return


    world[
        "characters"
    ].append({

        "name":
            character.name,

        "description":
            character.description,

        "sources":
            [source]
    })


# ============================================================
# LOCATION MERGE
# ============================================================

def add_location(
    world,
    location,
    source
):

    key = normalize(
        location.name
    )


    if not key:
        return


    for existing in world[
        "locations"
    ]:

        if normalize(
            existing.get(
                "name"
            )
        ) == key:

            merge_sources(
                existing.setdefault(
                    "sources",
                    []
                ),
                [source]
            )


            if (
                len(
                    location.description
                )
                >
                len(
                    existing.get(
                        "description",
                        ""
                    )
                )
            ):

                existing[
                    "description"
                ] = location.description


            return


    world[
        "locations"
    ].append({

        "name":
            location.name,

        "description":
            location.description,

        "sources":
            [source]
    })


# ============================================================
# RELATIONSHIP MERGE
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


    for existing in world[
        "relationships"
    ]:

        existing_key = (

            normalize(
                existing.get(
                    "subject"
                )
            ),

            normalize(
                existing.get(
                    "relation"
                )
            ),

            normalize(
                existing.get(
                    "object"
                )
            )
        )


        if existing_key == key:

            merge_sources(
                existing.setdefault(
                    "sources",
                    []
                ),
                [source]
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

        "sources":
            [source]
    })


# ============================================================
# EVENT MERGE
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


    for existing in world[
        "events"
    ]:

        if normalize(
            existing.get(
                "summary"
            )
        ) == key:

            merge_sources(
                existing.setdefault(
                    "sources",
                    []
                ),
                [source]
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

        "sources":
            [source]
    })


# ============================================================
# FACT MERGE
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


    for existing in world[
        "facts"
    ]:

        existing_key = (

            normalize(
                existing.get(
                    "subject"
                )
            ),

            normalize(
                existing.get(
                    "predicate"
                )
            ),

            normalize(
                existing.get(
                    "object"
                )
            )
        )


        if existing_key == key:

            merge_sources(
                existing.setdefault(
                    "sources",
                    []
                ),
                [source]
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

        "sources":
            [source]
    })


# ============================================================
# MERGE COMPLETE EXTRACTION
# ============================================================

def merge_extraction(
    world,
    extraction,
    chunk
):

    source = make_source(
        chunk
    )


    for item in extraction.characters:

        add_character(
            world,
            item,
            source
        )


    for item in extraction.locations:

        add_location(
            world,
            item,
            source
        )


    for item in extraction.relationships:

        add_relationship(
            world,
            item,
            source
        )


    for item in extraction.events:

        add_event(
            world,
            item,
            source
        )


    for item in extraction.facts:

        add_fact(
            world,
            item,
            source
        )


# ============================================================
# EXTRACTION PROMPT
# ============================================================

def build_prompt(
    chunk
):

    return f"""
You are extracting structured World Memory from ONE
passage of a fictional story.

Use ONLY the supplied story passage.

Do NOT use outside knowledge about:
- Sherlock Holmes
- the novel
- the author
- other chapters
- later events
- adaptations


============================================================
SOURCE
============================================================

Chunk ID:
{chunk.get("chunk_id")}

Chapter:
{chunk.get("chapter")}

Pages:
{chunk.get("page_start")} - {chunk.get("page_end")}


============================================================
STORY PASSAGE
============================================================

{chunk.get("text")}


============================================================
TASK
============================================================

Extract useful structured information that is actually
supported by this passage.


CHARACTERS
----------

Extract actual characters mentioned or clearly involved.

For each:
- name
- concise description

Prefer proper names when the passage establishes them.

Avoid creating separate character entities for vague
labels such as:

- Narrator
- The Speaker
- Watson's companion
- The Man
- His Companion

when the passage clearly identifies that person as an
already named character.


LOCATIONS
---------

Extract meaningful story-relevant locations.

For each:
- name
- concise description


RELATIONSHIPS
-------------

Extract meaningful relationships.

Use short snake_case labels such as:

friend_of
companion_of
works_with
knows
enemy_of
associated_with
in_touch_with
helped
warned
sent_message_to
speaks_to
employs
suspects
lives_at
visited
author_of


Only extract relationships actually supported by the
passage.


EVENTS
------

Extract important actions or developments that happen
in this passage.

Examples:

- receiving a letter
- decoding a message
- someone arriving
- discussing a threat
- discovering information
- committing a crime
- warning another character

Do NOT list every tiny physical action.

Extract events useful for answering:

"What happened?"
"What did this character do?"
"What happened before or after something?"


FACTS
-----

Extract useful facts established in the passage.

Possible predicates include:

identity
occupation
appearance
residence
reputation
status
personality
possesses
believes
knows
ability
role
motive


A fact differs from an event.

Example:

EVENT:
Holmes examines Porlock's letter.

FACT:
Holmes recognizes Porlock's handwriting.


============================================================
IMPORTANT RULES
============================================================

1. Use ONLY this passage.

2. Do not invent information.

3. Do not omit Facts merely because the same information
   appears in a character description.

4. Do not omit Events merely because an action also creates
   a relationship.

5. Quality is more important than quantity.

6. Keep descriptions and explanations concise.

7. Every category must be returned.

8. Empty lists are allowed only when that category truly
   has no useful information in this passage.
"""


# ============================================================
# RETRY PARSER
# ============================================================

def get_retry_seconds(
    error_message
):

    patterns = [

        r"retry in ([0-9.]+)s",

        r"retry after ([0-9.]+)s"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            error_message,
            re.IGNORECASE
        )


        if match:

            return (
                float(
                    match.group(1)
                )
                + 2
            )


    return 30


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_world_from_chunk(
    chunk
):

    prompt = build_prompt(
        chunk
    )


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            interaction = (
                client
                .interactions
                .create(

                    model=
                        LLM_MODEL,

                    input=
                        prompt,

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


            return (
                WorldExtraction
                .model_validate_json(
                    interaction.output_text
                )
            )


        except Exception as error:

            error_text = str(
                error
            )


            print()

            print(
                f"Gemini attempt "
                f"{attempt}/{MAX_RETRIES} failed."
            )

            print(
                error_text
            )


            if attempt >= MAX_RETRIES:

                raise


            if "429" in error_text:

                wait_seconds = (
                    get_retry_seconds(
                        error_text
                    )
                )

            else:

                wait_seconds = 5


            print(
                f"Retrying in "
                f"{wait_seconds:.0f}s..."
            )


            time.sleep(
                wait_seconds
            )


# ============================================================
# WORLD STATISTICS
# ============================================================

def print_stats(
    world
):

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

    print(
        f"Processed     : "
        f"{len(world['processed_chunk_ids'])}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "STORYWORLD — FULL WORLD MEMORY BUILDER"
    )

    print("=" * 70)


    chunks = load_chunks()

    world = load_world()


    processed_ids = {

        int(chunk_id)

        for chunk_id
        in world.get(
            "processed_chunk_ids",
            []
        )
    }


    # ========================================================
    # FIND ONLY UNPROCESSED CHUNKS
    # ========================================================

    remaining_chunks = [

        chunk

        for chunk
        in chunks

        if int(
            chunk.get(
                "chunk_id"
            )
        )
        not in processed_ids
    ]


    # ========================================================
    # OPTIONAL RUN LIMIT
    # ========================================================

    if MAX_CHUNKS is not None:

        remaining_chunks = (
            remaining_chunks[
                :MAX_CHUNKS
            ]
        )


    print()

    print(
        f"Total story chunks : "
        f"{len(chunks)}"
    )

    print(
        f"Already processed  : "
        f"{len(processed_ids)}"
    )

    print(
        f"Remaining this run : "
        f"{len(remaining_chunks)}"
    )


    # ========================================================
    # ALREADY COMPLETE
    # ========================================================

    if not remaining_chunks:

        save_world(
            world
        )

        print()
        print(
            "All story chunks have "
            "already been processed."
        )

        print_stats(
            world
        )

        return


    # ========================================================
    # PROCESS FULL BOOK
    # ========================================================

    for number, chunk in enumerate(
        remaining_chunks,
        start=1
    ):

        chunk_id = int(
            chunk[
                "chunk_id"
            ]
        )


        print()
        print("-" * 70)

        print(
            f"[{number}/"
            f"{len(remaining_chunks)}] "
            f"Processing chunk "
            f"{chunk_id}"
        )

        print(
            f"Chapter: "
            f"{chunk.get('chapter')}"
        )

        print(
            f"Pages: "
            f"{chunk.get('page_start')} "
            f"→ "
            f"{chunk.get('page_end')}"
        )


        try:

            extraction = (
                extract_world_from_chunk(
                    chunk
                )
            )


            # ------------------------------------------------
            # Useful diagnostic
            # ------------------------------------------------

            print(
                "✓ World information extracted"
            )

            print(
                f"  Characters    : "
                f"{len(extraction.characters)}"
            )

            print(
                f"  Locations     : "
                f"{len(extraction.locations)}"
            )

            print(
                f"  Relationships : "
                f"{len(extraction.relationships)}"
            )

            print(
                f"  Events        : "
                f"{len(extraction.events)}"
            )

            print(
                f"  Facts         : "
                f"{len(extraction.facts)}"
            )


            # ------------------------------------------------
            # Merge into persistent World Memory
            # ------------------------------------------------

            merge_extraction(
                world,
                extraction,
                chunk
            )


            # ------------------------------------------------
            # Mark processed ONLY after successful extraction
            # ------------------------------------------------

            if (
                chunk_id
                not in
                world[
                    "processed_chunk_ids"
                ]
            ):

                world[
                    "processed_chunk_ids"
                ].append(
                    chunk_id
                )


            world[
                "processed_chunk_ids"
            ] = sorted(
                set(
                    world[
                        "processed_chunk_ids"
                    ]
                )
            )


            # ------------------------------------------------
            # CHECKPOINT AFTER EVERY CHUNK
            # ------------------------------------------------

            save_progress(
                world
            )


            print_stats(
                world
            )


        except KeyboardInterrupt:

            print()
            print()
            print(
                "Run interrupted."
            )

            print(
                "Progress has been saved "
                "through the previous chunk."
            )

            return


        except Exception as error:

            print()
            print()

            print(
                "ERROR while processing "
                f"chunk {chunk_id}:"
            )

            print(
                error
            )

            print()

            print(
                "Progress has been preserved."
            )

            print(
                "Run python world_builder.py "
                "again to resume."
            )

            return


        if (
            number
            <
            len(
                remaining_chunks
            )
        ):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    save_progress(
        world
    )

    save_world(
        world
    )


    print()
    print()
    print("=" * 70)

    print(
        "FULL STORY WORLD MEMORY BUILD COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Saved:"
    )

    print(
        WORLD_FILE
    )

    print(
        PROGRESS_FILE
    )


    print_stats(
        world
    )


if __name__ == "__main__":

    main()