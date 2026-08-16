import json
from pathlib import Path
from typing import List, Literal

from google import genai
from pydantic import BaseModel


# ============================================================
# SETTINGS
# ============================================================

INPUT_WORLD = Path(
    "data/world_resolved.json"
)

OUTPUT_WORLD = Path(
    "data/world_semantic_resolved.json"
)

SEMANTIC_ALIASES_FILE = Path(
    "data/semantic_entity_aliases.json"
)

LLM_MODEL = "gemini-3.6-flash"


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class MergeGroup(BaseModel):

    canonical: str

    aliases: List[str]

    confidence: Literal[
        "high",
        "medium",
        "low"
    ]

    reason: str


class MergePlan(BaseModel):

    groups: List[MergeGroup]


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
        text.lower()
        .strip()
        .split()
    )


# ============================================================
# LOAD WORLD
# ============================================================

def load_world():

    if not INPUT_WORLD.exists():

        raise FileNotFoundError(
            f"{INPUT_WORLD} does not exist."
        )

    with open(
        INPUT_WORLD,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# BUILD CONTEXT FOR GEMINI
# ============================================================

def build_character_context(world):

    lines = []

    lines.append(
        "CHARACTERS:"
    )

    for character in world.get(
        "characters",
        []
    ):

        lines.append(
            f"""
NAME:
{character.get("name", "")}

DESCRIPTION:
{character.get("description", "")}
"""
        )


    lines.append(
        "\nRELATIONSHIPS:"
    )


    for relationship in world.get(
        "relationships",
        []
    ):

        lines.append(
            f"""
{relationship.get("subject", "")}
--{relationship.get("relation", "")}-->
{relationship.get("object", "")}

Evidence explanation:
{relationship.get("explanation", "")}
"""
        )


    lines.append(
        "\nFACTS:"
    )


    for fact in world.get(
        "facts",
        []
    ):

        lines.append(
            f"""
{fact.get("subject", "")}
--{fact.get("predicate", "")}-->
{fact.get("object", "")}

Evidence explanation:
{fact.get("explanation", "")}
"""
        )


    return "\n".join(
        lines
    )


# ============================================================
# ASK GEMINI FOR SEMANTIC DUPLICATES
# ============================================================

def create_merge_plan(world):

    context = build_character_context(
        world
    )

    existing_names = [
        character["name"]
        for character in world.get(
            "characters",
            []
        )
    ]

    prompt = f"""
You are resolving duplicate character entities extracted
from a fictional book.

Your job is NOT to add new story knowledge.

You must determine whether multiple extracted character
records actually refer to the SAME PERSON.

EXISTING CHARACTER NAMES:

{json.dumps(existing_names, indent=2)}


EXTRACTED WORLD INFORMATION:

{context}


RULES:

1. Only merge entities when they clearly represent the
   same individual.

2. Never merge two genuinely different people.

3. Role labels may be duplicates of named characters.

Examples:

"Narrator"
"The Speaker"
"Holmes's companion"
"Watson's companion"

These may refer to an already-named character, but ONLY
merge them if the supplied descriptions, relationships,
or facts make the identity clear.

4. A canonical name MUST be one of the existing character
   names supplied above.

5. Every alias MUST also exactly match one of the existing
   character names.

6. Prefer a proper personal name as canonical instead of
   vague labels such as:

Narrator
Speaker
Companion
Man
Woman
Visitor

7. Do NOT use outside knowledge about Sherlock Holmes or
the book.

Use ONLY the extracted information supplied here.

8. If an identity is uncertain, do NOT mark it high
confidence.

9. Only include groups that contain at least TWO names.

10. Confidence meanings:

high:
The extracted evidence makes them clearly the same person.

medium:
Likely the same, but some uncertainty exists.

low:
Possible but weak evidence.

Be conservative.
"""

    interaction = client.interactions.create(

        model=LLM_MODEL,

        input=prompt,

        response_format={
            "type":
                "text",

            "mime_type":
                "application/json",

            "schema":
                MergePlan.model_json_schema()
        },

        store=False
    )

    plan = MergePlan.model_validate_json(
        interaction.output_text
    )

    return plan


# ============================================================
# VALIDATE MERGE PLAN
# ============================================================

def build_alias_map(
    plan,
    world
):

    existing_names = {
        character["name"]
        for character in world.get(
            "characters",
            []
        )
    }

    alias_map = {}

    accepted_groups = []


    for group in plan.groups:

        # ----------------------------------------------------
        # Only HIGH CONFIDENCE merges happen automatically
        # ----------------------------------------------------

        if group.confidence != "high":

            continue


        if group.canonical not in existing_names:

            continue


        valid_aliases = []

        for alias in group.aliases:

            if alias in existing_names:

                valid_aliases.append(
                    alias
                )


        # Canonical should also resolve to itself
        if group.canonical not in valid_aliases:

            valid_aliases.append(
                group.canonical
            )


        valid_aliases = list(
            dict.fromkeys(
                valid_aliases
            )
        )


        if len(valid_aliases) < 2:

            continue


        # ----------------------------------------------------
        # Prevent one alias being assigned to different people
        # ----------------------------------------------------

        conflict = False

        for alias in valid_aliases:

            key = normalize(
                alias
            )

            if (
                key in alias_map
                and alias_map[key]
                != group.canonical
            ):

                conflict = True


        if conflict:

            print(
                "Skipping conflicting merge:",
                valid_aliases
            )

            continue


        for alias in valid_aliases:

            alias_map[
                normalize(alias)
            ] = group.canonical


        accepted_groups.append({
            "canonical":
                group.canonical,

            "aliases":
                valid_aliases,

            "confidence":
                group.confidence,

            "reason":
                group.reason
        })


    return (
        alias_map,
        accepted_groups
    )


# ============================================================
# RESOLVE A CHARACTER NAME
# ============================================================

def resolve_name(
    value,
    aliases
):

    if not value:

        return value

    return aliases.get(
        normalize(value),
        value
    )


# ============================================================
# MERGE SOURCES
# ============================================================

def merge_sources(
    target,
    incoming
):

    for source in incoming:

        if source not in target:

            target.append(
                source
            )


# ============================================================
# MERGE CHARACTERS
# ============================================================

def resolve_characters(
    characters,
    aliases
):

    merged = {}


    for character in characters:

        original = character[
            "name"
        ]

        canonical = resolve_name(
            original,
            aliases
        )


        if canonical not in merged:

            merged[
                canonical
            ] = {

                "name":
                    canonical,

                "description":
                    character.get(
                        "description",
                        ""
                    ),

                "sources":
                    list(
                        character.get(
                            "sources",
                            []
                        )
                    )
            }


        else:

            existing = merged[
                canonical
            ]


            merge_sources(
                existing[
                    "sources"
                ],

                character.get(
                    "sources",
                    []
                )
            )


            new_description = (
                character.get(
                    "description",
                    ""
                )
            )


            if len(
                new_description
            ) > len(
                existing[
                    "description"
                ]
            ):

                existing[
                    "description"
                ] = new_description


    return list(
        merged.values()
    )


# ============================================================
# RESOLVE RELATIONSHIPS
# ============================================================

def resolve_relationships(
    relationships,
    aliases
):

    merged = {}


    for relationship in relationships:

        subject = resolve_name(
            relationship.get(
                "subject",
                ""
            ),
            aliases
        )

        obj = resolve_name(
            relationship.get(
                "object",
                ""
            ),
            aliases
        )

        relation = relationship.get(
            "relation",
            ""
        )


        key = (
            normalize(subject),
            normalize(relation),
            normalize(obj)
        )


        if key not in merged:

            merged[key] = {

                "subject":
                    subject,

                "relation":
                    relation,

                "object":
                    obj,

                "explanation":
                    relationship.get(
                        "explanation",
                        ""
                    ),

                "sources":
                    list(
                        relationship.get(
                            "sources",
                            []
                        )
                    )
            }


        else:

            merge_sources(
                merged[key][
                    "sources"
                ],

                relationship.get(
                    "sources",
                    []
                )
            )


    return list(
        merged.values()
    )


# ============================================================
# RESOLVE FACTS
# ============================================================

def resolve_facts(
    facts,
    aliases
):

    merged = {}


    for fact in facts:

        subject = resolve_name(
            fact.get(
                "subject",
                ""
            ),
            aliases
        )

        obj = resolve_name(
            fact.get(
                "object",
                ""
            ),
            aliases
        )

        predicate = fact.get(
            "predicate",
            ""
        )


        key = (
            normalize(subject),
            normalize(predicate),
            normalize(obj)
        )


        if key not in merged:

            merged[key] = {

                "subject":
                    subject,

                "predicate":
                    predicate,

                "object":
                    obj,

                "explanation":
                    fact.get(
                        "explanation",
                        ""
                    ),

                "sources":
                    list(
                        fact.get(
                            "sources",
                            []
                        )
                    )
            }


        else:

            merge_sources(
                merged[key][
                    "sources"
                ],

                fact.get(
                    "sources",
                    []
                )
            )


    return list(
        merged.values()
    )


# ============================================================
# RESOLVE EVENTS
# ============================================================

def resolve_events(
    events,
    aliases
):

    resolved = []


    for event in events:

        participants = []


        for participant in event.get(
            "participants",
            []
        ):

            canonical = resolve_name(
                participant,
                aliases
            )

            if canonical not in participants:

                participants.append(
                    canonical
                )


        resolved.append({

            "summary":
                event.get(
                    "summary",
                    ""
                ),

            "participants":
                participants,

            # locations are NOT character-resolved
            "location":
                event.get(
                    "location",
                    ""
                ),

            "explanation":
                event.get(
                    "explanation",
                    ""
                ),

            "sources":
                event.get(
                    "sources",
                    []
                )
        })


    return resolved


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STORYWORLD — SEMANTIC ENTITY RESOLVER"
    )
    print("=" * 70)


    world = load_world()


    print()
    print(
        f"Characters before: "
        f"{len(world['characters'])}"
    )


    print()
    print(
        "Asking Gemini to identify semantic duplicates..."
    )


    plan = create_merge_plan(
        world
    )


    print()
    print("=" * 70)
    print("GEMINI MERGE PROPOSALS")
    print("=" * 70)


    if not plan.groups:

        print()
        print(
            "No semantic duplicate groups proposed."
        )


    for group in plan.groups:

        print()

        print(
            f"Canonical : "
            f"{group.canonical}"
        )

        print(
            f"Aliases   : "
            f"{group.aliases}"
        )

        print(
            f"Confidence: "
            f"{group.confidence}"
        )

        print(
            f"Reason    : "
            f"{group.reason}"
        )


    (
        aliases,
        accepted_groups
    ) = build_alias_map(
        plan,
        world
    )


    print()
    print("=" * 70)
    print("AUTOMATIC HIGH-CONFIDENCE MERGES")
    print("=" * 70)


    if not accepted_groups:

        print()
        print(
            "No high-confidence semantic merges applied."
        )


    for group in accepted_groups:

        print()

        print(
            f"{group['aliases']}"
        )

        print(
            "→"
        )

        print(
            group[
                "canonical"
            ]
        )


    # ========================================================
    # BUILD FINAL SEMANTICALLY RESOLVED WORLD
    # ========================================================

    resolved_world = {

        "processed_chunk_ids":
            world.get(
                "processed_chunk_ids",
                []
            ),

        "characters":
            resolve_characters(
                world[
                    "characters"
                ],
                aliases
            ),

        "locations":
            world[
                "locations"
            ],

        "relationships":
            resolve_relationships(
                world[
                    "relationships"
                ],
                aliases
            ),

        "events":
            resolve_events(
                world[
                    "events"
                ],
                aliases
            ),

        "facts":
            resolve_facts(
                world[
                    "facts"
                ],
                aliases
            )
    }


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_WORLD,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            resolved_world,
            file,
            ensure_ascii=False,
            indent=2
        )


    with open(
        SEMANTIC_ALIASES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "all_proposals": [
                    group.model_dump()
                    for group
                    in plan.groups
                ],

                "accepted_merges":
                    accepted_groups
            },

            file,
            ensure_ascii=False,
            indent=2
        )


    print()
    print("=" * 70)
    print("SEMANTIC ENTITY RESOLUTION COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Characters before : "
        f"{len(world['characters'])}"
    )

    print(
        f"Characters after  : "
        f"{len(resolved_world['characters'])}"
    )

    print()

    print(
        f"Saved:"
    )

    print(
        OUTPUT_WORLD
    )

    print(
        SEMANTIC_ALIASES_FILE
    )


if __name__ == "__main__":

    main()