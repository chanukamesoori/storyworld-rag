import argparse
import json
import re
import time
from pathlib import Path
from typing import List

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# GLOBAL SETTINGS
# ============================================================

CHUNKS_FILE = Path("data/chunks.json")

LLM_MODEL = "gemini-3.6-flash"

REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 3


# ============================================================
# STORY SCHEMA
# ============================================================

class StoryCharacter(BaseModel):
    name: str
    description: str


class StoryLocation(BaseModel):
    name: str
    description: str


class StoryRelationship(BaseModel):
    subject: str
    relation: str
    object: str
    explanation: str


class StoryEvent(BaseModel):
    summary: str
    participants: List[str] = Field(
        default_factory=list
    )
    location: str
    explanation: str


class StoryFact(BaseModel):
    subject: str
    predicate: str
    object: str
    explanation: str


class StoryExtraction(BaseModel):
    characters: List[StoryCharacter] = Field(
        default_factory=list
    )

    locations: List[StoryLocation] = Field(
        default_factory=list
    )

    relationships: List[StoryRelationship] = Field(
        default_factory=list
    )

    events: List[StoryEvent] = Field(
        default_factory=list
    )

    facts: List[StoryFact] = Field(
        default_factory=list
    )


# ============================================================
# TEXTBOOK SCHEMA
# ============================================================

class TextbookConcept(BaseModel):
    name: str
    explanation: str


class TextbookDefinition(BaseModel):
    term: str
    definition: str


class TextbookFormula(BaseModel):
    name: str
    formula: str
    variables: List[str] = Field(
        default_factory=list
    )
    explanation: str


class TextbookProcess(BaseModel):
    name: str
    steps: List[str] = Field(
        default_factory=list
    )
    explanation: str


class TextbookExample(BaseModel):
    topic: str
    summary: str


class ConceptRelationship(BaseModel):
    subject: str
    relation: str
    object: str
    explanation: str


class TextbookFact(BaseModel):
    subject: str
    predicate: str
    object: str
    explanation: str


class TextbookExtraction(BaseModel):
    concepts: List[TextbookConcept] = Field(
        default_factory=list
    )

    definitions: List[TextbookDefinition] = Field(
        default_factory=list
    )

    formulas: List[TextbookFormula] = Field(
        default_factory=list
    )

    processes: List[TextbookProcess] = Field(
        default_factory=list
    )

    examples: List[TextbookExample] = Field(
        default_factory=list
    )

    relationships: List[ConceptRelationship] = Field(
        default_factory=list
    )

    facts: List[TextbookFact] = Field(
        default_factory=list
    )


# ============================================================
# DOCUMENT PROFILES
# ============================================================

PROFILES = {
    "story": {
        "schema": StoryExtraction,

        "categories": [
            "characters",
            "locations",
            "relationships",
            "events",
            "facts"
        ]
    },

    "textbook": {
        "schema": TextbookExtraction,

        "categories": [
            "concepts",
            "definitions",
            "formulas",
            "processes",
            "examples",
            "relationships",
            "facts"
        ]
    }
}


# ============================================================
# GEMINI
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
# SOURCE METADATA
# ============================================================

def create_source(chunk):

    return {
        "chunk_id": chunk.get(
            "chunk_id"
        ),

        "chapter": chunk.get(
            "chapter",
            "Unknown"
        ),

        "page_start": chunk.get(
            "page_start",
            chunk.get("page")
        ),

        "page_end": chunk.get(
            "page_end",
            chunk.get("page")
        )
    }


# ============================================================
# STORY PROMPT
# ============================================================

def story_prompt(chunk):

    return f"""
You are extracting structured knowledge from a fictional
story.

Use ONLY the passage supplied below.

Do not use outside knowledge about:
- the book
- the author
- the characters
- other chapters
- adaptations


============================================================
SOURCE
============================================================

Chunk:
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
EXTRACT
============================================================

Extract only useful information actually established by
this passage.


CHARACTERS

Extract actual characters.

Prefer proper names.

Do NOT create unnecessary separate characters for vague
references such as:

"the narrator"
"the speaker"
"the man"
"his companion"

when the passage clearly identifies the person.


LOCATIONS

Extract meaningful named or story-relevant places.


RELATIONSHIPS

Extract useful relationships between entities.

Use short snake_case relationship names such as:

friend_of
companion_of
works_with
knows
married_to
enemy_of
in_touch_with
helped
warned
sent_message_to
employs
lives_at
suspects


EVENTS

Extract important events or actions.

Do not extract every tiny action.


FACTS

Extract useful facts about things such as:

identity
occupation
appearance
residence
reputation
status
belief
knowledge
possession
motive


IMPORTANT:

Do not invent information.

Quality is more important than quantity.
"""


# ============================================================
# TEXTBOOK PROMPT
# ============================================================

def textbook_prompt(chunk):

    return f"""
You are extracting structured educational knowledge from
a textbook or study document.

Use ONLY the passage supplied below.

Do not add outside knowledge.

Do not correct the textbook using your own knowledge.

Preserve the meaning taught by the supplied document.


============================================================
SOURCE
============================================================

Chunk:
{chunk.get("chunk_id")}

Section:
{chunk.get("chapter")}

Pages:
{chunk.get("page_start")} - {chunk.get("page_end")}


============================================================
TEXTBOOK PASSAGE
============================================================

{chunk.get("text")}


============================================================
EXTRACT
============================================================

Extract useful knowledge that would help a student later
understand the document and answer questions from it.


CONCEPTS

Extract important concepts or topics.

Example:

Net Present Value

Risk

Binary Search Tree

TCP Congestion Control


DEFINITIONS

Extract explicit or clearly established definitions.

Example:

term:
Opportunity Cost

definition:
The value of the next best alternative forgone.


FORMULAS

Extract formulas when actually present.

Include:

- formula name
- formula itself
- variables
- explanation

Do NOT invent formulas.


PROCESSES

Extract procedures, methods, algorithms, or sequences.

Examples:

Steps in hypothesis testing

AVL insertion procedure

TCP three-way handshake


EXAMPLES

Extract useful examples used by the material to explain
a concept.


RELATIONSHIPS

Extract meaningful relationships between concepts.

Use short snake_case labels such as:

uses
depends_on
part_of
causes
measured_by
calculated_using
example_of
contrasts_with
leads_to
requires


FACTS

Extract other important stable facts taught by the
material.


IMPORTANT:

Do not create information that is not present.

Do not use general knowledge to fill gaps.

Quality is more important than quantity.
"""


# ============================================================
# SELECT PROMPT
# ============================================================

def build_prompt(
    document_type,
    chunk
):

    if document_type == "story":

        return story_prompt(
            chunk
        )

    if document_type == "textbook":

        return textbook_prompt(
            chunk
        )

    raise ValueError(
        f"Unsupported document type: "
        f"{document_type}"
    )


# ============================================================
# CREATE EMPTY MEMORY
# ============================================================

def create_empty_memory(
    document_type
):

    memory = {
        "document_type":
            document_type,

        "processed_chunk_ids":
            []
    }


    for category in (
        PROFILES[
            document_type
        ][
            "categories"
        ]
    ):

        memory[
            category
        ] = []


    return memory


# ============================================================
# OUTPUT PATHS
# ============================================================

def get_paths(
    document_type
):

    progress_file = Path(
        f"data/"
        f"structured_{document_type}_progress.json"
    )

    final_file = Path(
        f"data/"
        f"structured_{document_type}.json"
    )

    return (
        progress_file,
        final_file
    )


# ============================================================
# LOAD MEMORY
# ============================================================

def load_memory(
    document_type,
    progress_file
):

    if not progress_file.exists():

        return create_empty_memory(
            document_type
        )


    with open(
        progress_file,
        "r",
        encoding="utf-8"
    ) as file:

        memory = json.load(
            file
        )


    if (
        memory.get(
            "document_type"
        )
        != document_type
    ):

        raise RuntimeError(
            "Progress file document type "
            "does not match."
        )


    return memory


# ============================================================
# SAVE
# ============================================================

def save_json(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# RETRY PARSER
# ============================================================

def get_retry_seconds(
    error_message
):

    match = re.search(
        r"retry in ([0-9.]+)s",
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


    return 10


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_chunk(
    document_type,
    chunk
):

    profile = PROFILES[
        document_type
    ]

    schema = profile[
        "schema"
    ]

    prompt = build_prompt(
        document_type,
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
                            schema
                            .model_json_schema()
                    },

                    store=False
                )
            )


            return (
                schema
                .model_validate_json(
                    interaction.output_text
                )
            )


        except Exception as error:

            print()

            print(
                f"Attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed:"
            )

            print(
                error
            )


            if attempt == MAX_RETRIES:

                raise


            error_text = str(
                error
            )


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
# DEDUPLICATION KEYS
# ============================================================

def get_item_key(
    document_type,
    category,
    item
):

    # ========================================================
    # STORY
    # ========================================================

    if document_type == "story":

        if category == "characters":

            return normalize(
                item.get(
                    "name"
                )
            )


        if category == "locations":

            return normalize(
                item.get(
                    "name"
                )
            )


        if category == "relationships":

            return (
                normalize(
                    item.get(
                        "subject"
                    )
                ),
                normalize(
                    item.get(
                        "relation"
                    )
                ),
                normalize(
                    item.get(
                        "object"
                    )
                )
            )


        if category == "events":

            return normalize(
                item.get(
                    "summary"
                )
            )


        if category == "facts":

            return (
                normalize(
                    item.get(
                        "subject"
                    )
                ),
                normalize(
                    item.get(
                        "predicate"
                    )
                ),
                normalize(
                    item.get(
                        "object"
                    )
                )
            )


    # ========================================================
    # TEXTBOOK
    # ========================================================

    if document_type == "textbook":

        if category == "concepts":

            return normalize(
                item.get(
                    "name"
                )
            )


        if category == "definitions":

            return normalize(
                item.get(
                    "term"
                )
            )


        if category == "formulas":

            return (
                normalize(
                    item.get(
                        "name"
                    )
                ),
                normalize(
                    item.get(
                        "formula"
                    )
                )
            )


        if category == "processes":

            return normalize(
                item.get(
                    "name"
                )
            )


        if category == "examples":

            return (
                normalize(
                    item.get(
                        "topic"
                    )
                ),
                normalize(
                    item.get(
                        "summary"
                    )
                )
            )


        if category == "relationships":

            return (
                normalize(
                    item.get(
                        "subject"
                    )
                ),
                normalize(
                    item.get(
                        "relation"
                    )
                ),
                normalize(
                    item.get(
                        "object"
                    )
                )
            )


        if category == "facts":

            return (
                normalize(
                    item.get(
                        "subject"
                    )
                ),
                normalize(
                    item.get(
                        "predicate"
                    )
                ),
                normalize(
                    item.get(
                        "object"
                    )
                )
            )


    return json.dumps(
        item,
        sort_keys=True
    )


# ============================================================
# MERGE SOURCES
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

        if (
            source.get(
                "chunk_id"
            )
            not in existing_ids
        ):

            existing.append(
                source
            )


# ============================================================
# MERGE EXTRACTION
# ============================================================

def merge_extraction(
    memory,
    extraction,
    document_type,
    chunk
):

    source = create_source(
        chunk
    )


    categories = (
        PROFILES[
            document_type
        ][
            "categories"
        ]
    )


    extraction_dict = (
        extraction.model_dump()
    )


    for category in categories:

        new_items = (
            extraction_dict.get(
                category,
                []
            )
        )


        for item in new_items:

            item[
                "sources"
            ] = [
                source
            ]


            new_key = get_item_key(
                document_type,
                category,
                item
            )


            matched = False


            for existing in memory[
                category
            ]:

                existing_key = (
                    get_item_key(
                        document_type,
                        category,
                        existing
                    )
                )


                if existing_key == new_key:

                    merge_sources(
                        existing.setdefault(
                            "sources",
                            []
                        ),

                        item[
                            "sources"
                        ]
                    )

                    matched = True

                    break


            if not matched:

                memory[
                    category
                ].append(
                    item
                )


# ============================================================
# DISPLAY MEMORY STATS
# ============================================================

def print_stats(
    memory,
    document_type
):

    print()

    print(
        "Current Structured Memory:"
    )


    for category in (
        PROFILES[
            document_type
        ][
            "categories"
        ]
    ):

        print(
            f"{category.title():15}: "
            f"{len(memory[category])}"
        )


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            "data/chunks.json not found. "
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Build structured memory for a document."
    )


    parser.add_argument(
        "--type",
        choices=[
            "story",
            "textbook"
        ],
        required=True,
        help=
        "Type of document being processed."
    )


    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=
        "Optional number of chunks to process. "
        "Omit to process all chunks."
    )


    parser.add_argument(
        "--fresh",
        action="store_true",
        help=
        "Ignore previous progress and rebuild."
    )


    args = parser.parse_args()


    document_type = args.type


    (
        progress_file,
        final_file
    ) = get_paths(
        document_type
    )


    print()
    print("=" * 70)

    print(
        "DOCUMENT INTELLIGENCE — "
        "STRUCTURED MEMORY BUILDER"
    )

    print("=" * 70)

    print()

    print(
        f"Document type : "
        f"{document_type}"
    )

    print(
        f"Model         : "
        f"{LLM_MODEL}"
    )


    chunks = load_chunks()


    if args.fresh:

        memory = create_empty_memory(
            document_type
        )

    else:

        memory = load_memory(
            document_type,
            progress_file
        )


    processed_ids = set(
        memory.get(
            "processed_chunk_ids",
            []
        )
    )


    remaining_chunks = [

        chunk

        for chunk
        in chunks

        if chunk.get(
            "chunk_id"
        )
        not in processed_ids
    ]


    if args.max_chunks is not None:

        remaining_chunks = (
            remaining_chunks[
                :args.max_chunks
            ]
        )


    print()

    print(
        f"Available chunks : "
        f"{len(chunks)}"
    )

    print(
        f"Already processed: "
        f"{len(processed_ids)}"
    )

    print(
        f"This run         : "
        f"{len(remaining_chunks)}"
    )


    if not remaining_chunks:

        print()
        print(
            "Nothing left to process."
        )

        return


    # ========================================================
    # PROCESS
    # ========================================================

    for number, chunk in enumerate(
        remaining_chunks,
        start=1
    ):

        print()
        print("-" * 70)

        print(
            f"[{number}/"
            f"{len(remaining_chunks)}]"
        )

        print(
            f"Chunk   : "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"Section : "
            f"{chunk.get('chapter')}"
        )

        print(
            f"Pages   : "
            f"{chunk.get('page_start')} "
            f"→ "
            f"{chunk.get('page_end')}"
        )


        try:

            extraction = (
                extract_chunk(
                    document_type,
                    chunk
                )
            )


            merge_extraction(
                memory,
                extraction,
                document_type,
                chunk
            )


            chunk_id = chunk.get(
                "chunk_id"
            )


            if (
                chunk_id
                not in memory[
                    "processed_chunk_ids"
                ]
            ):

                memory[
                    "processed_chunk_ids"
                ].append(
                    chunk_id
                )


            save_json(
                progress_file,
                memory
            )


            print(
                "✓ Structured knowledge extracted"
            )


            print_stats(
                memory,
                document_type
            )


        except Exception as error:

            print()
            print(
                "ERROR while processing chunk:"
            )

            print(
                error
            )

            print()

            print(
                "Progress before this chunk "
                "has already been saved."
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
    # SAVE FINAL
    # ========================================================

    save_json(
        final_file,
        memory
    )


    print()
    print("=" * 70)

    print(
        "STRUCTURED MEMORY BUILD COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Document type:"
        f" {document_type}"
    )

    print(
        f"Saved:"
    )

    print(
        final_file
    )


    print_stats(
        memory,
        document_type
    )


if __name__ == "__main__":

    main()