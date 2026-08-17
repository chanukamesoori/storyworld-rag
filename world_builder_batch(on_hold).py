"""
this part is made because i was using the gemini free api and if i hit the limit save the
current progress and move from there onece it is free to use the api again but i paid the gemini api
so im holding the developmet of this part
"""

import json
import re
import time
from pathlib import Path
from typing import List

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# SETTINGS
# ============================================================

CHUNKS_FILE = Path(
    "data/chunks.json"
)

PROGRESS_FILE = Path(
    "data/world_full_progress.json"
)

FINAL_WORLD_FILE = Path(
    "data/world_full_raw.json"
)

BATCH_AUDIT_DIR = Path(
    "data/world_batches"
)

BATCH_AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LLM_MODEL = "gemini-3.6-flash"


# ------------------------------------------------------------
# 326 chunks / 18 ≈ 19 Gemini requests
#
# This is deliberately much cheaper than:
# 326 chunks = 326 requests
# ------------------------------------------------------------

BATCH_SIZE = 18


# Small delay helps with requests-per-minute limits.
REQUEST_DELAY_SECONDS = 4


# If Gemini says wait 30-60 seconds, automatically retry.
#
# If it says wait for a very long period, save progress
# and exit instead of leaving the terminal sleeping for hours.
MAX_AUTOMATIC_WAIT_SECONDS = 90


MAX_RETRIES = 3


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class CharacterExtraction(BaseModel):

    name: str

    description: str

    source_chunk_ids: List[int] = Field(
        default_factory=list
    )


class LocationExtraction(BaseModel):

    name: str

    description: str

    source_chunk_ids: List[int] = Field(
        default_factory=list
    )


class RelationshipExtraction(BaseModel):

    subject: str

    relation: str

    object: str

    explanation: str

    source_chunk_ids: List[int] = Field(
        default_factory=list
    )


class EventExtraction(BaseModel):

    summary: str

    participants: List[str]

    location: str

    explanation: str

    source_chunk_ids: List[int] = Field(
        default_factory=list
    )


class FactExtraction(BaseModel):

    subject: str

    predicate: str

    object: str

    explanation: str

    source_chunk_ids: List[int] = Field(
        default_factory=list
    )


class WorldBatchExtraction(BaseModel):

    characters: List[
        CharacterExtraction
    ] = Field(
        default_factory=list
    )

    locations: List[
        LocationExtraction
    ] = Field(
        default_factory=list
    )

    relationships: List[
        RelationshipExtraction
    ] = Field(
        default_factory=list
    )

    events: List[
        EventExtraction
    ] = Field(
        default_factory=list
    )

    facts: List[
        FactExtraction
    ] = Field(
        default_factory=list
    )


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
        text
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
            f"{CHUNKS_FILE} not found."
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
# LOAD PROGRESS
# ============================================================

def load_progress():

    if not PROGRESS_FILE.exists():

        return create_empty_world()


    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


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
# SAVE FINAL WORLD
# ============================================================

def save_final_world(world):

    with open(
        FINAL_WORLD_FILE,
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
# CHUNK LOOKUP
# ============================================================

def create_chunk_lookup(
    chunks
):

    return {

        int(
            chunk["chunk_id"]
        ):
        chunk

        for chunk
        in chunks
    }


# ============================================================
# SOURCE OBJECT
# ============================================================

def source_from_chunk(
    chunk
):

    return {

        "chunk_id":
            int(
                chunk[
                    "chunk_id"
                ]
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
# VALIDATE SOURCE IDS
# ============================================================

def get_sources(
    source_chunk_ids,
    allowed_chunk_ids,
    chunk_lookup
):

    sources = []

    seen = set()


    for chunk_id in source_chunk_ids:

        try:

            chunk_id = int(
                chunk_id
            )

        except Exception:

            continue


        if chunk_id not in allowed_chunk_ids:

            continue


        if chunk_id in seen:

            continue


        if chunk_id not in chunk_lookup:

            continue


        seen.add(
            chunk_id
        )


        sources.append(
            source_from_chunk(
                chunk_lookup[
                    chunk_id
                ]
            )
        )


    return sources


# ============================================================
# MERGE SOURCE LISTS
# ============================================================

def merge_sources(
    existing,
    incoming
):

    existing_ids = {

        source.get(
            "chunk_id"
        )

        for source
        in existing
    }


    for source in incoming:

        chunk_id = source.get(
            "chunk_id"
        )

        if chunk_id not in existing_ids:

            existing.append(
                source
            )

            existing_ids.add(
                chunk_id
            )


# ============================================================
# BUILD PROMPT TEXT
# ============================================================

def build_batch_text(
    batch
):

    sections = []


    for chunk in batch:

        chunk_id = chunk[
            "chunk_id"
        ]

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

        text = chunk.get(
            "text",
            ""
        )


        sections.append(
            f"""
============================================================
CHUNK {chunk_id}
============================================================

CHAPTER:
{chapter}

PAGES:
{page_start} - {page_end}

STORY TEXT:

{text}
"""
        )


    return "\n".join(
        sections
    )


# ============================================================
# EXTRACTION PROMPT
# ============================================================

def build_prompt(
    batch
):

    batch_text = build_batch_text(
        batch
    )


    valid_ids = [

        int(
            chunk[
                "chunk_id"
            ]
        )

        for chunk
        in batch
    ]


    return f"""
You are constructing structured World Memory from
passages of a fictional story.

Use ONLY the supplied story passages.

Do NOT use:
- prior knowledge of Sherlock Holmes
- later knowledge of the novel
- adaptations
- outside knowledge
- assumptions that are not established by these passages


============================================================
AVAILABLE CHUNK IDS
============================================================

{valid_ids}


============================================================
STORY PASSAGES
============================================================

{batch_text}


============================================================
EXTRACTION TASK
============================================================

Extract useful structured world information from these
passages.


CHARACTERS
----------

Extract actual people or clearly identified characters.

For each character provide:

- name
- concise description
- source_chunk_ids


Do NOT create separate characters merely because the
story uses:

- "he"
- "she"
- "I"
- "the narrator"
- "the speaker"
- "the man"
- "his companion"

when the passage clearly identifies that role as an
already named person.

Prefer the actual proper name when the supplied passages
make that identity explicit.


LOCATIONS
---------

Extract meaningful named or story-relevant locations.

Avoid vague incidental descriptions unless the place has
narrative importance.


RELATIONSHIPS
-------------

Extract useful relationships.

Examples of relation labels:

friend_of
companion_of
works_with
knows
married_to
parent_of
enemy_of
associated_with
in_touch_with
helped
warned
sent_message_to
speaks_to
suspects
employs
lives_at
visited

Use a short normalized snake_case relation.

Do not extract meaningless relationships merely because
two characters appear in the same paragraph.


EVENTS
------

Extract important actions or events.

Keep event summaries concise.

Include:

- participants
- location if established
- explanation
- source_chunk_ids


FACTS
-----

Extract useful stable facts about:

- identity
- occupation
- appearance
- residence
- reputation
- possessions
- beliefs
- knowledge
- status
- motives when explicitly established
- other significant story information

Use a short normalized predicate.


============================================================
VERY IMPORTANT SOURCE RULE
============================================================

Every extracted item MUST contain source_chunk_ids.

Only use IDs from:

{valid_ids}

Include every supplied chunk that directly supports the
item.

Never invent a source chunk ID.


============================================================
DEDUPLICATION
============================================================

These passages overlap slightly.

Do NOT repeat the same fact/event/relationship simply
because it appears in multiple overlapping chunks.

Merge repeated evidence into one item and include all
supporting source_chunk_ids.


============================================================
QUALITY OVER QUANTITY
============================================================

Extract information that will actually help answer
questions about the fictional world.

Avoid trivial observations.

Keep descriptions and explanations concise.
"""


# ============================================================
# RETRY TIME PARSER
# ============================================================

def get_retry_seconds(
    error_text
):

    patterns = [

        r"retry in ([0-9.]+)s",

        r"retry after ([0-9.]+)s"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            error_text,
            re.IGNORECASE
        )


        if match:

            return (
                float(
                    match.group(1)
                )
                + 2
            )


    return 60


# ============================================================
# CALL GEMINI
# ============================================================

def extract_batch(
    batch
):

    prompt = build_prompt(
        batch
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
                            WorldBatchExtraction
                            .model_json_schema()
                    },

                    store=False
                )
            )


            return (
                WorldBatchExtraction
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


                if (
                    wait_seconds
                    > MAX_AUTOMATIC_WAIT_SECONDS
                ):

                    raise RuntimeError(
                        "LONG_QUOTA_WAIT:"
                        f"{wait_seconds}"
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
# MERGE CHARACTER
# ============================================================

def merge_character(
    world,
    item,
    sources
):

    name = item.name.strip()

    if not name:
        return


    key = normalize(
        name
    )


    for existing in world[
        "characters"
    ]:

        if normalize(
            existing[
                "name"
            ]
        ) == key:

            merge_sources(
                existing[
                    "sources"
                ],
                sources
            )


            if (
                len(
                    item.description
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
                ] = item.description


            return


    world[
        "characters"
    ].append({

        "name":
            name,

        "description":
            item.description,

        "sources":
            sources
    })


# ============================================================
# MERGE LOCATION
# ============================================================

def merge_location(
    world,
    item,
    sources
):

    name = item.name.strip()

    if not name:
        return


    key = normalize(
        name
    )


    for existing in world[
        "locations"
    ]:

        if normalize(
            existing[
                "name"
            ]
        ) == key:

            merge_sources(
                existing[
                    "sources"
                ],
                sources
            )


            if (
                len(
                    item.description
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
                ] = item.description


            return


    world[
        "locations"
    ].append({

        "name":
            name,

        "description":
            item.description,

        "sources":
            sources
    })


# ============================================================
# MERGE RELATIONSHIP
# ============================================================

def merge_relationship(
    world,
    item,
    sources
):

    key = (

        normalize(
            item.subject
        ),

        normalize(
            item.relation
        ),

        normalize(
            item.object
        )
    )


    for existing in world[
        "relationships"
    ]:

        existing_key = (

            normalize(
                existing[
                    "subject"
                ]
            ),

            normalize(
                existing[
                    "relation"
                ]
            ),

            normalize(
                existing[
                    "object"
                ]
            )
        )


        if existing_key == key:

            merge_sources(
                existing[
                    "sources"
                ],
                sources
            )

            return


    world[
        "relationships"
    ].append({

        "subject":
            item.subject,

        "relation":
            item.relation,

        "object":
            item.object,

        "explanation":
            item.explanation,

        "sources":
            sources
    })


# ============================================================
# MERGE EVENT
# ============================================================

def merge_event(
    world,
    item,
    sources
):

    key = normalize(
        item.summary
    )


    for existing in world[
        "events"
    ]:

        if normalize(
            existing[
                "summary"
            ]
        ) == key:

            merge_sources(
                existing[
                    "sources"
                ],
                sources
            )

            return


    world[
        "events"
    ].append({

        "summary":
            item.summary,

        "participants":
            item.participants,

        "location":
            item.location,

        "explanation":
            item.explanation,

        "sources":
            sources
    })


# ============================================================
# MERGE FACT
# ============================================================

def merge_fact(
    world,
    item,
    sources
):

    key = (

        normalize(
            item.subject
        ),

        normalize(
            item.predicate
        ),

        normalize(
            item.object
        )
    )


    for existing in world[
        "facts"
    ]:

        existing_key = (

            normalize(
                existing[
                    "subject"
                ]
            ),

            normalize(
                existing[
                    "predicate"
                ]
            ),

            normalize(
                existing[
                    "object"
                ]
            )
        )


        if existing_key == key:

            merge_sources(
                existing[
                    "sources"
                ],
                sources
            )

            return


    world[
        "facts"
    ].append({

        "subject":
            item.subject,

        "predicate":
            item.predicate,

        "object":
            item.object,

        "explanation":
            item.explanation,

        "sources":
            sources
    })


# ============================================================
# MERGE COMPLETE EXTRACTION
# ============================================================

def merge_extraction(
    world,
    extraction,
    batch,
    chunk_lookup
):

    allowed_ids = {

        int(
            chunk[
                "chunk_id"
            ]
        )

        for chunk
        in batch
    }


    for item in extraction.characters:

        sources = get_sources(
            item.source_chunk_ids,
            allowed_ids,
            chunk_lookup
        )

        merge_character(
            world,
            item,
            sources
        )


    for item in extraction.locations:

        sources = get_sources(
            item.source_chunk_ids,
            allowed_ids,
            chunk_lookup
        )

        merge_location(
            world,
            item,
            sources
        )


    for item in extraction.relationships:

        sources = get_sources(
            item.source_chunk_ids,
            allowed_ids,
            chunk_lookup
        )

        merge_relationship(
            world,
            item,
            sources
        )


    for item in extraction.events:

        sources = get_sources(
            item.source_chunk_ids,
            allowed_ids,
            chunk_lookup
        )

        merge_event(
            world,
            item,
            sources
        )


    for item in extraction.facts:

        sources = get_sources(
            item.source_chunk_ids,
            allowed_ids,
            chunk_lookup
        )

        merge_fact(
            world,
            item,
            sources
        )


# ============================================================
# SAVE RAW BATCH FOR AUDITING
# ============================================================

def save_batch_audit(
    extraction,
    batch
):

    first_id = int(
        batch[0][
            "chunk_id"
        ]
    )

    last_id = int(
        batch[-1][
            "chunk_id"
        ]
    )


    path = (
        BATCH_AUDIT_DIR
        /
        f"batch_{first_id:04d}_{last_id:04d}.json"
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            extraction.model_dump(),
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# WORLD STATISTICS
# ============================================================

def print_stats(
    world
):

    print()
    print(
        "Current Full World Memory:"
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
        f"Chunks done   : "
        f"{len(world['processed_chunk_ids'])}"
    )


# ============================================================
# CREATE REMAINING BATCHES
# ============================================================

def create_remaining_batches(
    chunks,
    processed_ids
):

    remaining = [

        chunk

        for chunk
        in chunks

        if int(
            chunk[
                "chunk_id"
            ]
        )
        not in processed_ids
    ]


    return [

        remaining[
            index:
            index + BATCH_SIZE
        ]

        for index
        in range(
            0,
            len(remaining),
            BATCH_SIZE
        )
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STORYWORLD — FULL BOOK BATCH WORLD BUILDER"
    )
    print("=" * 70)


    chunks = load_chunks()

    chunk_lookup = (
        create_chunk_lookup(
            chunks
        )
    )


    world = load_progress()


    processed_ids = {

        int(chunk_id)

        for chunk_id
        in world.get(
            "processed_chunk_ids",
            []
        )
    }


    batches = (
        create_remaining_batches(
            chunks,
            processed_ids
        )
    )


    print()
    print(
        f"Story chunks       : "
        f"{len(chunks)}"
    )

    print(
        f"Already processed  : "
        f"{len(processed_ids)}"
    )

    print(
        f"Remaining chunks   : "
        f"{len(chunks) - len(processed_ids)}"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gemini calls needed: "
        f"{len(batches)}"
    )


    # --------------------------------------------------------
    # Nothing remaining
    # --------------------------------------------------------

    if not batches:

        save_final_world(
            world
        )

        print()
        print(
            "All chunks are already processed."
        )

        print_stats(
            world
        )

        return


    # ========================================================
    # PROCESS BATCHES
    # ========================================================

    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        first_id = batch[
            0
        ][
            "chunk_id"
        ]

        last_id = batch[
            -1
        ][
            "chunk_id"
        ]


        print()
        print()
        print("-" * 70)

        print(
            f"BATCH "
            f"{batch_number}/{len(batches)}"
        )

        print(
            f"Chunks: "
            f"{first_id} → {last_id}"
        )

        print(
            f"Pages: "
            f"{batch[0].get('page_start')} "
            f"→ "
            f"{batch[-1].get('page_end')}"
        )

        print("-" * 70)


        try:

            extraction = (
                extract_batch(
                    batch
                )
            )


        except Exception as error:

            error_text = str(
                error
            )


            save_progress(
                world
            )


            print()
            print(
                "Batch stopped."
            )


            if error_text.startswith(
                "LONG_QUOTA_WAIT:"
            ):

                seconds = (
                    error_text.split(
                        ":",
                        1
                    )[1]
                )

                print(
                    "Gemini returned a long "
                    f"quota wait ({seconds}s)."
                )

                print(
                    "Progress has been saved."
                )

                print(
                    "Run the same command again "
                    "after your Gemini quota "
                    "becomes available."
                )

                return


            print(
                "ERROR:"
            )

            print(
                error
            )

            print()
            print(
                "Progress has been saved."
            )

            print(
                "Rerunning this script will "
                "resume from this batch."
            )

            return


        # ----------------------------------------------------
        # Audit raw model extraction
        # ----------------------------------------------------

        save_batch_audit(
            extraction,
            batch
        )


        # ----------------------------------------------------
        # Merge into World Memory
        # ----------------------------------------------------

        merge_extraction(
            world,
            extraction,
            batch,
            chunk_lookup
        )


        # ----------------------------------------------------
        # Mark ALL batch chunks processed
        # only AFTER successful extraction + merge
        # ----------------------------------------------------

        for chunk in batch:

            chunk_id = int(
                chunk[
                    "chunk_id"
                ]
            )


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


        # Keep IDs ordered
        world[
            "processed_chunk_ids"
        ] = sorted(
            set(
                world[
                    "processed_chunk_ids"
                ]
            )
        )


        # ----------------------------------------------------
        # SAVE AFTER EVERY SUCCESSFUL BATCH
        # ----------------------------------------------------

        save_progress(
            world
        )


        print()
        print(
            "✓ Batch extracted and checkpointed"
        )


        print(
            f"Extracted:"
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


        print_stats(
            world
        )


        if batch_number < len(
            batches
        ):

            time.sleep(
                REQUEST_DELAY_SECONDS
            )


    # ========================================================
    # ALL CHUNKS COMPLETE
    # ========================================================

    save_final_world(
        world
    )


    print()
    print()
    print("=" * 70)

    print(
        "FULL BOOK WORLD MEMORY COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Saved raw full world:"
    )

    print(
        FINAL_WORLD_FILE
    )

    print()

    print(
        f"Checkpoint:"
    )

    print(
        PROGRESS_FILE
    )

    print()

    print(
        f"Individual batch audits:"
    )

    print(
        BATCH_AUDIT_DIR
    )


    print_stats(
        world
    )


if __name__ == "__main__":

    main()