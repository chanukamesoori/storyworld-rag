import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

WORLD_FILE = Path(
    "data/world.json"
)

RESOLVED_WORLD_FILE = Path(
    "data/world_resolved.json"
)

ALIASES_FILE = Path(
    "data/entity_aliases.json"
)


# ============================================================
# COMMON TITLES
# ============================================================

TITLES = {
    "mr",
    "mrs",
    "miss",
    "ms",
    "dr",
    "doctor",
    "prof",
    "professor",
    "sir",
    "lady",
    "lord",
    "inspector",
    "detective",
    "colonel",
    "captain",
    "major",
    "sergeant",
    "constable",
    "reverend"
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^\w\s-]",
        " ",
        text
    )

    # Treat hyphens as spaces for matching
    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# REMOVE TITLES
# ============================================================

def remove_titles(text):

    normalized = normalize_text(
        text
    )

    words = normalized.split()

    cleaned = [
        word
        for word in words
        if word not in TITLES
    ]

    return " ".join(
        cleaned
    )


# ============================================================
# NAME TOKENS
# ============================================================

def name_tokens(name):

    return remove_titles(
        name
    ).split()


# ============================================================
# DETERMINE WHETHER TWO CHARACTER NAMES MATCH
# ============================================================

def basic_name_match(
    name1,
    name2
):

    n1 = normalize_text(
        name1
    )

    n2 = normalize_text(
        name2
    )

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if n1 == n2:
        return True


    # --------------------------------------------------------
    # Match after removing titles
    #
    # Mr. Holmes == Holmes
    # Professor Moriarty == Moriarty
    # --------------------------------------------------------

    stripped1 = remove_titles(
        name1
    )

    stripped2 = remove_titles(
        name2
    )

    if (
        stripped1
        and stripped1 == stripped2
    ):
        return True


    return False


# ============================================================
# BUILD CHARACTER GROUPS
# ============================================================

def build_character_groups(
    characters
):

    names = [
        item["name"]
        for item in characters
        if item.get("name")
    ]

    # --------------------------------------------------------
    # Start with basic exact/title matches
    # --------------------------------------------------------

    groups = []

    assigned = set()


    for i, name in enumerate(
        names
    ):

        if i in assigned:
            continue

        group = [
            name
        ]

        assigned.add(
            i
        )


        for j in range(
            i + 1,
            len(names)
        ):

            if j in assigned:
                continue

            other = names[j]

            if basic_name_match(
                name,
                other
            ):

                group.append(
                    other
                )

                assigned.add(
                    j
                )


        groups.append(
            group
        )


    # --------------------------------------------------------
    # Build surname index from the CURRENT groups
    #
    # Sherlock Holmes
    # Mr Holmes
    #
    # → "holmes"
    # --------------------------------------------------------

    full_name_candidates = (
        defaultdict(list)
    )


    for group_index, group in enumerate(
        groups
    ):

        all_tokens = []

        for name in group:

            tokens = name_tokens(
                name
            )

            if len(tokens) >= 2:

                all_tokens.append(
                    tokens
                )


        for tokens in all_tokens:

            surname = tokens[-1]

            full_name_candidates[
                surname
            ].append(
                group_index
            )


    # Remove duplicate group indexes
    for surname in list(
        full_name_candidates.keys()
    ):

        full_name_candidates[
            surname
        ] = list(
            set(
                full_name_candidates[
                    surname
                ]
            )
        )


    # --------------------------------------------------------
    # Merge unique one-word surname groups
    #
    # Holmes
    #   +
    # Sherlock Holmes
    #
    # ONLY when Holmes matches exactly ONE full-name group.
    # --------------------------------------------------------

    merges = {}


    for group_index, group in enumerate(
        groups
    ):

        if group_index in merges:
            continue


        one_word_names = []

        for name in group:

            tokens = name_tokens(
                name
            )

            if len(tokens) == 1:

                one_word_names.append(
                    tokens[0]
                )


        for surname in one_word_names:

            candidate_groups = (
                full_name_candidates.get(
                    surname,
                    []
                )
            )

            candidate_groups = [
                candidate
                for candidate in candidate_groups
                if candidate != group_index
            ]


            # IMPORTANT:
            # only merge when unambiguous
            if len(candidate_groups) == 1:

                target_group = (
                    candidate_groups[0]
                )

                merges[
                    group_index
                ] = target_group

                break


    # --------------------------------------------------------
    # Apply merges
    # --------------------------------------------------------

    final_groups = []

    consumed = set()


    for index, group in enumerate(
        groups
    ):

        if index in consumed:
            continue

        # This group belongs to another group
        if index in merges:
            continue


        combined = list(
            group
        )


        for source_index, target_index in (
            merges.items()
        ):

            if target_index == index:

                combined.extend(
                    groups[source_index]
                )

                consumed.add(
                    source_index
                )


        # Remove duplicates while preserving order
        combined = list(
            dict.fromkeys(
                combined
            )
        )

        final_groups.append(
            combined
        )


    # Any groups not included because of complex merge
    # relationships are preserved.
    included_names = {
        name
        for group in final_groups
        for name in group
    }


    for group in groups:

        if not any(
            name in included_names
            for name in group
        ):

            final_groups.append(
                group
            )


    return final_groups


# ============================================================
# CHOOSE BEST CANONICAL CHARACTER NAME
# ============================================================

def canonical_name_score(
    name
):

    tokens = name_tokens(
        name
    )

    raw_tokens = normalize_text(
        name
    ).split()


    # More actual name tokens = better
    score = (
        len(tokens) * 100
    )

    # Slight preference for longer/full names
    score += len(
        name
    )

    # If the first non-title token appears to be
    # more than a surname, reward it.
    if len(tokens) >= 2:
        score += 100

    # Titles can be useful but should not dominate
    if (
        raw_tokens
        and raw_tokens[0] in TITLES
    ):
        score += 5

    return score


def choose_canonical_name(
    names
):

    return max(
        names,
        key=canonical_name_score
    )


# ============================================================
# BUILD CHARACTER ALIAS MAP
# ============================================================

def build_character_alias_map(
    characters
):

    groups = build_character_groups(
        characters
    )

    alias_map = {}

    report = []


    for group in groups:

        canonical = (
            choose_canonical_name(
                group
            )
        )


        for alias in group:

            alias_map[
                normalize_text(alias)
            ] = canonical

            alias_map[
                remove_titles(alias)
            ] = canonical


        report.append({

            "canonical":
                canonical,

            "aliases":
                group
        })


    return alias_map, report


# ============================================================
# LOCATION ALIAS MAP
#
# Locations are intentionally handled conservatively.
# We do NOT assume:
#
# Baker Street == 221B Baker Street
#
# ============================================================

def build_location_alias_map(
    locations
):

    alias_map = {}

    groups = {}


    for location in locations:

        name = location.get(
            "name",
            ""
        )

        normalized = normalize_text(
            name
        )

        if not normalized:
            continue


        if normalized not in groups:

            groups[
                normalized
            ] = []

        groups[
            normalized
        ].append(
            name
        )


    report = []


    for normalized, names in (
        groups.items()
    ):

        # choose longest representation
        canonical = max(
            names,
            key=len
        )


        for alias in names:

            alias_map[
                normalize_text(alias)
            ] = canonical


        report.append({

            "canonical":
                canonical,

            "aliases":
                list(
                    dict.fromkeys(
                        names
                    )
                )
        })


    return alias_map, report


# ============================================================
# RESOLVE ENTITY NAME
# ============================================================

def resolve_entity(
    value,
    character_aliases,
    location_aliases
):

    if not value:
        return value


    normalized = normalize_text(
        value
    )

    stripped = remove_titles(
        value
    )


    # Characters first
    if normalized in character_aliases:

        return character_aliases[
            normalized
        ]


    if stripped in character_aliases:

        return character_aliases[
            stripped
        ]


    # Then locations
    if normalized in location_aliases:

        return location_aliases[
            normalized
        ]


    return value


# ============================================================
# MERGE SOURCE LISTS
# ============================================================

def merge_sources(
    destination,
    incoming
):

    for source in incoming:

        if source not in destination:

            destination.append(
                source
            )


# ============================================================
# RESOLVE CHARACTERS
# ============================================================

def resolve_characters(
    characters,
    aliases
):

    merged = {}


    for character in characters:

        original_name = (
            character["name"]
        )

        canonical = resolve_entity(
            original_name,
            aliases,
            {}
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

            existing = (
                merged[
                    canonical
                ]
            )

            merge_sources(
                existing["sources"],
                character.get(
                    "sources",
                    []
                )
            )


            # Prefer longer description
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
# RESOLVE LOCATIONS
# ============================================================

def resolve_locations(
    locations,
    aliases
):

    merged = {}


    for location in locations:

        name = location[
            "name"
        ]

        canonical = (
            aliases.get(
                normalize_text(name),
                name
            )
        )


        if canonical not in merged:

            merged[
                canonical
            ] = {

                "name":
                    canonical,

                "description":
                    location.get(
                        "description",
                        ""
                    ),

                "sources":
                    list(
                        location.get(
                            "sources",
                            []
                        )
                    )
            }

        else:

            merge_sources(
                merged[
                    canonical
                ]["sources"],

                location.get(
                    "sources",
                    []
                )
            )


    return list(
        merged.values()
    )


# ============================================================
# RESOLVE RELATIONSHIPS
# ============================================================

def resolve_relationships(
    relationships,
    character_aliases,
    location_aliases
):

    merged = {}


    for relationship in relationships:

        subject = resolve_entity(
            relationship["subject"],
            character_aliases,
            location_aliases
        )

        obj = resolve_entity(
            relationship["object"],
            character_aliases,
            location_aliases
        )

        relation = relationship[
            "relation"
        ]


        key = (

            normalize_text(subject),

            normalize_text(relation),

            normalize_text(obj)
        )


        if key not in merged:

            merged[
                key
            ] = {

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
                merged[
                    key
                ]["sources"],

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
    character_aliases,
    location_aliases
):

    merged = {}


    for fact in facts:

        subject = resolve_entity(
            fact["subject"],
            character_aliases,
            location_aliases
        )


        obj = resolve_entity(
            fact["object"],
            character_aliases,
            location_aliases
        )


        predicate = fact[
            "predicate"
        ]


        key = (

            normalize_text(subject),

            normalize_text(predicate),

            normalize_text(obj)
        )


        if key not in merged:

            merged[
                key
            ] = {

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
                merged[
                    key
                ]["sources"],

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
    character_aliases,
    location_aliases
):

    resolved_events = []


    for event in events:

        participants = []


        for participant in event.get(
            "participants",
            []
        ):

            resolved = resolve_entity(
                participant,
                character_aliases,
                location_aliases
            )

            if resolved not in participants:

                participants.append(
                    resolved
                )


        location = resolve_entity(
            event.get(
                "location",
                ""
            ),
            character_aliases,
            location_aliases
        )


        resolved_events.append({

            "summary":
                event.get(
                    "summary",
                    ""
                ),

            "participants":
                participants,

            "location":
                location,

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


    return resolved_events


# ============================================================
# PRINT ALIAS REPORT
# ============================================================

def print_alias_report(
    report,
    entity_type
):

    print()
    print("=" * 70)

    print(
        f"{entity_type.upper()} ENTITY RESOLUTION"
    )

    print("=" * 70)


    merged_count = 0


    for item in report:

        aliases = item[
            "aliases"
        ]

        if len(aliases) <= 1:
            continue


        merged_count += 1


        print()

        print(
            f"Canonical: "
            f"{item['canonical']}"
        )

        print(
            "Aliases:"
        )


        for alias in aliases:

            print(
                f"  - {alias}"
            )


    if merged_count == 0:

        print()
        print(
            "No duplicate aliases "
            "were automatically merged."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "STORYWORLD — ENTITY RESOLVER"
    )

    print("=" * 70)


    if not WORLD_FILE.exists():

        print()
        print(
            "ERROR:"
        )

        print(
            f"{WORLD_FILE} not found."
        )

        print(
            "Run world_builder.py first."
        )

        return


    # --------------------------------------------------------
    # LOAD WORLD
    # --------------------------------------------------------

    with open(
        WORLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        world = json.load(
            file
        )


    print()

    print(
        "Original World Memory:"
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
    # CHARACTER ALIASES
    # --------------------------------------------------------

    (
        character_aliases,
        character_report
    ) = build_character_alias_map(
        world[
            "characters"
        ]
    )


    # --------------------------------------------------------
    # LOCATION ALIASES
    # --------------------------------------------------------

    (
        location_aliases,
        location_report
    ) = build_location_alias_map(
        world[
            "locations"
        ]
    )


    print_alias_report(
        character_report,
        "Character"
    )


    print_alias_report(
        location_report,
        "Location"
    )


    # --------------------------------------------------------
    # BUILD RESOLVED WORLD
    # --------------------------------------------------------

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
                character_aliases
            ),

        "locations":
            resolve_locations(
                world[
                    "locations"
                ],
                location_aliases
            ),

        "relationships":
            resolve_relationships(
                world[
                    "relationships"
                ],
                character_aliases,
                location_aliases
            ),

        "events":
            resolve_events(
                world[
                    "events"
                ],
                character_aliases,
                location_aliases
            ),

        "facts":
            resolve_facts(
                world[
                    "facts"
                ],
                character_aliases,
                location_aliases
            )
    }


    # --------------------------------------------------------
    # SAVE RESOLVED WORLD
    # --------------------------------------------------------

    with open(
        RESOLVED_WORLD_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            resolved_world,
            file,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # SAVE ALIAS INFORMATION
    # --------------------------------------------------------

    aliases_output = {

        "characters":
            character_report,

        "locations":
            location_report
    }


    with open(
        ALIASES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            aliases_output,
            file,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "RESOLVED WORLD MEMORY"
    )

    print("=" * 70)

    print()

    print(
        f"Characters    : "
        f"{len(resolved_world['characters'])}"
    )

    print(
        f"Locations     : "
        f"{len(resolved_world['locations'])}"
    )

    print(
        f"Relationships : "
        f"{len(resolved_world['relationships'])}"
    )

    print(
        f"Events        : "
        f"{len(resolved_world['events'])}"
    )

    print(
        f"Facts         : "
        f"{len(resolved_world['facts'])}"
    )

    print()

    print(
        f"Saved resolved world:"
    )

    print(
        RESOLVED_WORLD_FILE
    )

    print()

    print(
        f"Saved entity aliases:"
    )

    print(
        ALIASES_FILE
    )


if __name__ == "__main__":

    main()