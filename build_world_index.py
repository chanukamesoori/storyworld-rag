import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

WORLD_FILE = Path(
    "data/world_semantic_resolved.json"
)

WORLD_ITEMS_FILE = Path(
    "data/world_items.json"
)

WORLD_EMBEDDINGS_FILE = Path(
    "data/world_embeddings.npy"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD WORLD
# ============================================================

def load_world():

    if not WORLD_FILE.exists():

        raise FileNotFoundError(
            f"{WORLD_FILE} does not exist. "
            "Run entity_resolver.py first."
        )

    with open(
        WORLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# BUILD SEARCHABLE CHARACTER ITEMS
# ============================================================

def build_character_items(
    world
):

    items = []

    for character in world.get(
        "characters",
        []
    ):

        name = character.get(
            "name",
            ""
        )

        description = character.get(
            "description",
            ""
        )

        text = (
            f"Character: {name}. "
            f"{description}"
        )

        items.append({

            "type":
                "character",

            "text":
                text,

            "name":
                name,

            "description":
                description,

            "sources":
                character.get(
                    "sources",
                    []
                )
        })

    return items


# ============================================================
# BUILD SEARCHABLE LOCATION ITEMS
# ============================================================

def build_location_items(
    world
):

    items = []

    for location in world.get(
        "locations",
        []
    ):

        name = location.get(
            "name",
            ""
        )

        description = location.get(
            "description",
            ""
        )

        text = (
            f"Location: {name}. "
            f"{description}"
        )

        items.append({

            "type":
                "location",

            "text":
                text,

            "name":
                name,

            "description":
                description,

            "sources":
                location.get(
                    "sources",
                    []
                )
        })

    return items


# ============================================================
# BUILD SEARCHABLE RELATIONSHIPS
# ============================================================

def build_relationship_items(
    world
):

    items = []

    for relationship in world.get(
        "relationships",
        []
    ):

        subject = relationship.get(
            "subject",
            ""
        )

        relation = relationship.get(
            "relation",
            ""
        )

        obj = relationship.get(
            "object",
            ""
        )

        explanation = relationship.get(
            "explanation",
            ""
        )

        text = (
            f"{subject} "
            f"{relation.replace('_', ' ')} "
            f"{obj}. "
            f"{explanation}"
        )

        items.append({

            "type":
                "relationship",

            "text":
                text,

            "subject":
                subject,

            "relation":
                relation,

            "object":
                obj,

            "explanation":
                explanation,

            "sources":
                relationship.get(
                    "sources",
                    []
                )
        })

    return items


# ============================================================
# BUILD SEARCHABLE FACTS
# ============================================================

def build_fact_items(
    world
):

    items = []

    for fact in world.get(
        "facts",
        []
    ):

        subject = fact.get(
            "subject",
            ""
        )

        predicate = fact.get(
            "predicate",
            ""
        )

        obj = fact.get(
            "object",
            ""
        )

        explanation = fact.get(
            "explanation",
            ""
        )

        text = (
            f"{subject} "
            f"{predicate.replace('_', ' ')} "
            f"{obj}. "
            f"{explanation}"
        )

        items.append({

            "type":
                "fact",

            "text":
                text,

            "subject":
                subject,

            "predicate":
                predicate,

            "object":
                obj,

            "explanation":
                explanation,

            "sources":
                fact.get(
                    "sources",
                    []
                )
        })

    return items


# ============================================================
# BUILD SEARCHABLE EVENTS
# ============================================================

def build_event_items(
    world
):

    items = []

    for event in world.get(
        "events",
        []
    ):

        summary = event.get(
            "summary",
            ""
        )

        participants = event.get(
            "participants",
            []
        )

        location = event.get(
            "location",
            ""
        )

        explanation = event.get(
            "explanation",
            ""
        )

        participant_text = ", ".join(
            participants
        )

        text = (
            f"Event: {summary}. "
            f"Participants: "
            f"{participant_text}. "
        )

        if location:

            text += (
                f"Location: {location}. "
            )

        text += explanation


        items.append({

            "type":
                "event",

            "text":
                text,

            "summary":
                summary,

            "participants":
                participants,

            "location":
                location,

            "explanation":
                explanation,

            "sources":
                event.get(
                    "sources",
                    []
                )
        })

    return items


# ============================================================
# BUILD COMPLETE WORLD INDEX
# ============================================================

def build_world_items(
    world
):

    items = []

    items.extend(
        build_character_items(
            world
        )
    )

    items.extend(
        build_location_items(
            world
        )
    )

    items.extend(
        build_relationship_items(
            world
        )
    )

    items.extend(
        build_fact_items(
            world
        )
    )

    items.extend(
        build_event_items(
            world
        )
    )


    # Assign IDs
    for index, item in enumerate(
        items
    ):

        item[
            "world_item_id"
        ] = index


    return items


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    items
):

    print()
    print(
        "Loading local embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    texts = [

        item["text"]

        for item
        in items
    ]


    print(
        f"Creating embeddings for "
        f"{len(texts)} world-memory items..."
    )


    embeddings = model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=True
    )


    print(
        f"World embedding matrix: "
        f"{embeddings.shape}"
    )


    return embeddings


# ============================================================
# SAVE
# ============================================================

def save_world_index(
    items,
    embeddings
):

    with open(
        WORLD_ITEMS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            items,
            file,
            ensure_ascii=False,
            indent=2
        )


    np.save(
        WORLD_EMBEDDINGS_FILE,
        embeddings
    )


# ============================================================
# PREVIEW
# ============================================================

def preview(
    items,
    amount=10
):

    print()
    print("=" * 70)
    print("WORLD MEMORY PREVIEW")
    print("=" * 70)


    for item in items[
        :amount
    ]:

        print()

        print(
            f"ID   : "
            f"{item['world_item_id']}"
        )

        print(
            f"TYPE : "
            f"{item['type']}"
        )

        print(
            f"TEXT : "
            f"{item['text']}"
        )

        print(
            "-" * 70
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "STORYWORLD — WORLD INDEX BUILDER"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load resolved World Memory
    # --------------------------------------------------------

    world = load_world()


    print()
    print(
        "Loaded resolved world:"
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


    # --------------------------------------------------------
    # Convert World Memory to searchable documents
    # --------------------------------------------------------

    items = build_world_items(
        world
    )


    print()
    print(
        f"Created "
        f"{len(items)} searchable "
        f"World Memory items."
    )


    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    preview(
        items
    )


    # --------------------------------------------------------
    # Create LOCAL embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        items
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_world_index(
        items,
        embeddings
    )


    print()
    print("=" * 70)

    print(
        "WORLD MEMORY INDEX CREATED"
    )

    print("=" * 70)

    print()

    print(
        f"Saved:"
    )

    print(
        WORLD_ITEMS_FILE
    )

    print(
        WORLD_EMBEDDINGS_FILE
    )


if __name__ == "__main__":

    main()