import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


# ============================================================
# FILES
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
# TITLES
# ============================================================

TITLE_ALIASES = {
    "mr": "mr",
    "mister": "mr",

    "mrs": "mrs",

    "miss": "miss",
    "ms": "ms",

    "dr": "dr",
    "doctor": "dr",

    "prof": "professor",
    "professor": "professor",

    "sir": "sir",
    "lady": "lady",
    "lord": "lord",

    "inspector": "inspector",
    "detective": "detective",

    "colonel": "colonel",
    "captain": "captain",
    "major": "major",
    "sergeant": "sergeant",
    "constable": "constable",

    "reverend": "reverend"
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SPLIT TITLE FROM NAME
# ============================================================

def split_title(name):

    normalized = normalize_text(
        name
    )

    words = normalized.split()

    if not words:

        return None, []


    first = words[0]


    if first in TITLE_ALIASES:

        title = TITLE_ALIASES[
            first
        ]

        return (
            title,
            words[1:]
        )


    return (
        None,
        words
    )


# ============================================================
# CORE NAME
# ============================================================

def core_name(name):

    _, tokens = split_title(
        name
    )

    return " ".join(
        tokens
    )


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(
        self,
        size
    ):

        self.parent = list(
            range(size)
        )


    def find(
        self,
        value
    ):

        while (
            self.parent[value]
            != value
        ):

            self.parent[value] = (
                self.parent[
                    self.parent[value]
                ]
            )

            value = self.parent[
                value
            ]


        return value


    def union(
        self,
        first,
        second
    ):

        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )


        if root_first != root_second:

            self.parent[
                root_second
            ] = root_first


# ============================================================
# SAME CORE NAME CHECK
# ============================================================

def safely_same_core(
    name1,
    name2,
    titled_variants_by_core
):

    title1, tokens1 = split_title(
        name1
    )

    title2, tokens2 = split_title(
        name2
    )


    if not tokens1 or not tokens2:

        return False


    if tokens1 != tokens2:

        return False


    # --------------------------------------------------------
    # Exactly same full representation
    # --------------------------------------------------------

    if normalize_text(
        name1
    ) == normalize_text(
        name2
    ):

        return True


    # --------------------------------------------------------
    # Same title:
    #
    # Dr Roylott
    # Doctor Roylott
    # --------------------------------------------------------

    if (
        title1
        and title2
        and title1 == title2
    ):

        return True


    # --------------------------------------------------------
    # DIFFERENT titles:
    #
    # Dr Roylott
    # Miss Roylott
    #
    # NEVER merge deterministically.
    # --------------------------------------------------------

    if (
        title1
        and title2
        and title1 != title2
    ):

        return False


    # --------------------------------------------------------
    # One has title and one does not.
    #
    # Full names are normally safe:
    #
    # Sherlock Holmes
    # Mr Sherlock Holmes
    # --------------------------------------------------------

    if len(
        tokens1
    ) >= 2:

        return True


    # --------------------------------------------------------
    # Single surname is dangerous.
    #
    # Holmes + Mr Holmes can merge if Mr is the ONLY
    # titled form for "Holmes".
    #
    # Roylott + Dr Roylott + Miss Roylott cannot.
    # --------------------------------------------------------

    core = " ".join(
        tokens1
    )


    distinct_titles = (
        titled_variants_by_core.get(
            core,
            set()
        )
    )


    if len(
        distinct_titles
    ) <= 1:

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

        for item
        in characters

        if item.get(
            "name"
        )
    ]


    size = len(
        names
    )


    uf = UnionFind(
        size
    )


    # ========================================================
    # RECORD TITLED VARIANTS
    # ========================================================

    titled_variants_by_core = (
        defaultdict(set)
    )


    for name in names:

        title, tokens = split_title(
            name
        )

        if (
            title
            and tokens
        ):

            titled_variants_by_core[
                " ".join(tokens)
            ].add(
                title
            )


    # ========================================================
    # PASS 1
    # Exact / safe same-core matches
    # ========================================================

    for i in range(
        size
    ):

        for j in range(
            i + 1,
            size
        ):

            if safely_same_core(
                names[i],
                names[j],
                titled_variants_by_core
            ):

                uf.union(
                    i,
                    j
                )


    # ========================================================
    # BUILD FULL-NAME CANDIDATES BY SURNAME
    # ========================================================

    full_candidates = (
        defaultdict(list)
    )


    for index, name in enumerate(
        names
    ):

        title, tokens = split_title(
            name
        )


        if len(
            tokens
        ) >= 2:

            surname = (
                tokens[-1]
            )


            full_candidates[
                surname
            ].append({

                "index":
                    index,

                "title":
                    title,

                "tokens":
                    tokens,

                "name":
                    name
            })


    # ========================================================
    # PASS 2
    # Safe surname → full-name resolution
    # ========================================================

    for index, name in enumerate(
        names
    ):

        short_title, short_tokens = (
            split_title(
                name
            )
        )


        # Only surname-only forms
        if len(
            short_tokens
        ) != 1:

            continue


        surname = short_tokens[
            0
        ]


        candidates = (
            full_candidates.get(
                surname,
                []
            )
        )


        compatible = []


        for candidate in candidates:

            candidate_title = (
                candidate[
                    "title"
                ]
            )


            # -----------------------------------------------
            # Example:
            #
            # Dr Roylott
            # Dr Grimesby Roylott
            #
            # YES
            # -----------------------------------------------

            if short_title:

                if (
                    candidate_title
                    and
                    candidate_title
                    != short_title
                ):

                    continue


            compatible.append(
                candidate
            )


        # ----------------------------------------------------
        # Only merge if ONE clear full-name candidate exists.
        #
        # Miss Stoner:
        #
        # Helen Stoner
        # Julia Stoner
        #
        # => ambiguous, no merge
        # ----------------------------------------------------

        unique_roots = {

            uf.find(
                candidate[
                    "index"
                ]
            )

            for candidate
            in compatible
        }


        if len(
            unique_roots
        ) != 1:

            continue


        target_index = (
            compatible[
                0
            ][
                "index"
            ]
        )


        # ----------------------------------------------------
        # Extra protection for bare surnames.
        #
        # If multiple different titled surname variants exist,
        # do not guess which one a bare surname belongs to.
        # ----------------------------------------------------

        if short_title is None:

            titles = (
                titled_variants_by_core.get(
                    surname,
                    set()
                )
            )


            if len(
                titles
            ) > 1:

                continue


        uf.union(
            index,
            target_index
        )


    # ========================================================
    # COLLECT GROUPS
    # ========================================================

    grouped = (
        defaultdict(list)
    )


    for index, name in enumerate(
        names
    ):

        root = uf.find(
            index
        )

        grouped[
            root
        ].append(
            name
        )


    groups = []


    for group in grouped.values():

        group = list(
            dict.fromkeys(
                group
            )
        )

        groups.append(
            group
        )


    return groups


# ============================================================
# CANONICAL NAME SCORE
# ============================================================

def canonical_name_score(
    name
):

    title, tokens = split_title(
        name
    )


    score = 0


    # Full names strongly preferred.
    score += (
        len(tokens)
        * 100
    )


    # Prefer longer specific representations.
    score += len(
        name
    )


    # Proper names with 2+ tokens are best.
    if len(
        tokens
    ) >= 2:

        score += 200


    # Titles add a small amount.
    if title:

        score += 10


    return score


# ============================================================
# CHOOSE CANONICAL NAME
# ============================================================

def choose_canonical_name(
    names
):

    return max(
        names,
        key=canonical_name_score
    )


# ============================================================
# BUILD CHARACTER ALIASES
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


        # IMPORTANT:
        #
        # ONLY exact aliases are registered.
        #
        # We do NOT register title-stripped aliases.
        # That was the source of dangerous merges.
        # ----------------------------------------------------

        for alias in group:

            alias_map[
                normalize_text(
                    alias
                )
            ] = canonical


        report.append({

            "canonical":
                canonical,

            "aliases":
                group
        })


    return (
        alias_map,
        report
    )


# ============================================================
# LOCATIONS
#
# Deliberately conservative: exact matches only.
# ============================================================

def build_location_alias_map(
    locations
):

    alias_map = {}

    groups = defaultdict(
        list
    )


    for location in locations:

        name = location.get(
            "name",
            ""
        )


        key = normalize_text(
            name
        )


        if not key:

            continue


        groups[
            key
        ].append(
            name
        )


    report = []


    for key, names in groups.items():

        canonical = max(
            names,
            key=len
        )


        aliases = list(
            dict.fromkeys(
                names
            )
        )


        for alias in aliases:

            alias_map[
                normalize_text(
                    alias
                )
            ] = canonical


        report.append({

            "canonical":
                canonical,

            "aliases":
                aliases
        })


    return (
        alias_map,
        report
    )


# ============================================================
# RESOLVE ENTITY
# ============================================================

def resolve_entity(
    value,
    character_aliases,
    location_aliases
):

    if not value:

        return value


    key = normalize_text(
        value
    )


    if key in character_aliases:

        return character_aliases[
            key
        ]


    if key in location_aliases:

        return location_aliases[
            key
        ]


    return value


# ============================================================
# MERGE SOURCES
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

        original = character[
            "name"
        ]


        canonical = (
            aliases.get(
                normalize_text(
                    original
                ),
                original
            )
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


            if (
                len(
                    new_description
                )
                >
                len(
                    existing[
                        "description"
                    ]
                )
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
                normalize_text(
                    name
                ),
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
                ][
                    "sources"
                ],

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
            relationship[
                "subject"
            ],
            character_aliases,
            location_aliases
        )


        obj = resolve_entity(
            relationship[
                "object"
            ],
            character_aliases,
            location_aliases
        )


        relation = relationship[
            "relation"
        ]


        key = (

            normalize_text(
                subject
            ),

            normalize_text(
                relation
            ),

            normalize_text(
                obj
            )
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
                ][
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
    character_aliases,
    location_aliases
):

    merged = {}


    for fact in facts:

        subject = resolve_entity(
            fact[
                "subject"
            ],
            character_aliases,
            location_aliases
        )


        obj = resolve_entity(
            fact[
                "object"
            ],
            character_aliases,
            location_aliases
        )


        predicate = fact[
            "predicate"
        ]


        key = (

            normalize_text(
                subject
            ),

            normalize_text(
                predicate
            ),

            normalize_text(
                obj
            )
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
                ][
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
    character_aliases,
    location_aliases
):

    resolved = []


    for event in events:

        participants = []


        for participant in event.get(
            "participants",
            []
        ):

            canonical = resolve_entity(
                participant,
                character_aliases,
                location_aliases
            )


            if canonical not in participants:

                participants.append(
                    canonical
                )


        location = resolve_entity(
            event.get(
                "location",
                ""
            ),
            character_aliases,
            location_aliases
        )


        resolved.append({

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


    return resolved


# ============================================================
# PRINT REPORT
# ============================================================

def print_alias_report(
    report,
    entity_type
):

    print()
    print("=" * 70)

    print(
        f"{entity_type.upper()} "
        f"ENTITY RESOLUTION"
    )

    print("=" * 70)


    merged_count = 0


    for item in report:

        aliases = item[
            "aliases"
        ]


        if len(
            aliases
        ) <= 1:

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
            "No safe automatic merges found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "STORYWORLD — CONSERVATIVE "
        "ENTITY RESOLVER"
    )

    print("=" * 70)


    if not WORLD_FILE.exists():

        print()

        print(
            f"ERROR: "
            f"{WORLD_FILE} not found."
        )

        return


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


    (
        character_aliases,
        character_report
    ) = build_character_alias_map(
        world[
            "characters"
        ]
    )


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


    with open(
        ALIASES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "characters":
                    character_report,

                "locations":
                    location_report
            },
            file,
            ensure_ascii=False,
            indent=2
        )


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
        "Saved resolved world:"
    )

    print(
        RESOLVED_WORLD_FILE
    )


if __name__ == "__main__":

    main()