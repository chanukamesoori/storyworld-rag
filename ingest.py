from pathlib import Path
import json
import re

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer


# ============================================================
# SETTINGS
# ============================================================

BOOK_PATH = Path("books/sherlock.pdf")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_WORDS = 220
MAX_WORDS = 300
OVERLAP_PARAGRAPHS = 1


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert repeated whitespace/newlines to normal spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PRIMARY STORY HEADING
# ============================================================

def is_primary_story_heading(text):

    """
    Used to find where the actual story begins.

    Examples:
    PART I
    PART I.
    BOOK II
    CHAPTER 1
    CHAPTER I.
    CHAPTER II. Sherlock Holmes Discourses
    """

    text = clean_text(text)

    if not text:
        return False

    if len(text) > 150:
        return False

    return bool(
        re.match(
            r"^(chapter|part|book)\s+"
            r"([0-9]+|[ivxlcdm]+)"
            r"\.?(?:\s+.*)?$",
            text,
            re.IGNORECASE
        )
    )


# ============================================================
# NORMAL CHAPTER HEADING DETECTION
# ============================================================

def is_chapter_heading(text):

    text = clean_text(text)

    if not text:
        return False

    if len(text) > 150:
        return False

    # --------------------------------------------------------
    # CHAPTER I
    # CHAPTER II. Sherlock Holmes Discourses
    # PART I
    # BOOK II
    # --------------------------------------------------------

    if is_primary_story_heading(text):
        return True

    # --------------------------------------------------------
    # I. A SCANDAL IN BOHEMIA
    # II. THE RED-HEADED LEAGUE
    # --------------------------------------------------------

    if re.match(
        r"^[IVXLCDM]+\.\s+[A-Za-z]",
        text
    ):
        return True

    # --------------------------------------------------------
    # ALL-CAPS title
    #
    # THE VALLEY OF FEAR       ✓
    # 26 BIRLSTONE 9 47 171    ✗
    # --------------------------------------------------------

    words = text.split()

    if (
        2 <= len(words) <= 10
        and text.upper() == text
        and not any(
            char.isdigit()
            for char in text
        )
        and any(
            char.isalpha()
            for char in text
        )
    ):
        return True

    return False


# ============================================================
# DETECT TABLE OF CONTENTS PAGE
# ============================================================

def is_toc_page(page_text):

    """
    Detect pages that look like a table of contents.

    Typical examples:

    The Warning ................. 5
    Sherlock Holmes Discourses .. 9
    The Tragedy of Birlstone .... 12
    """

    if not page_text:
        return False

    raw_lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    text_lower = page_text.lower()


    # --------------------------------------------------------
    # Explicit Contents heading
    # --------------------------------------------------------

    if (
        "table of contents" in text_lower
        or re.search(
            r"\bcontents\b",
            text_lower
        )
    ):
        return True


    # --------------------------------------------------------
    # Count dotted leaders:
    #
    # Chapter title ............ 10
    # --------------------------------------------------------

    dotted_lines = 0

    for line in raw_lines:

        if re.search(
            r"\.{4,}",
            line
        ):
            dotted_lines += 1


    if dotted_lines >= 2:
        return True


    # --------------------------------------------------------
    # Detect many short entries ending with page numbers
    # --------------------------------------------------------

    numbered_lines = 0

    for line in raw_lines:

        if re.search(
            r"\s\d{1,4}\s*$",
            line
        ):
            numbered_lines += 1


    if numbered_lines >= 5:
        return True


    return False


# ============================================================
# READ PDF INTO PAGE BLOCKS
# ============================================================

def read_pdf_pages(path):

    print()
    print("Reading book...")

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        raw_page_text = page.get_text(
            "text",
            sort=True
        )

        raw_blocks = page.get_text(
            "blocks",
            sort=True
        )

        blocks = []

        for block in raw_blocks:

            text = clean_text(
                block[4]
            )

            if not text:
                continue

            # Ignore plain page-number blocks
            if re.fullmatch(
                r"\d+",
                text
            ):
                continue

            blocks.append(
                text
            )

        pages.append({

            "page":
                page_number,

            "raw_text":
                raw_page_text,

            "blocks":
                blocks,

            "is_toc":
                is_toc_page(
                    raw_page_text
                )
        })

    document.close()

    print(
        f"Read {len(pages)} PDF pages."
    )

    return pages


# ============================================================
# FIND WHERE REAL STORY STARTS
# ============================================================

def find_story_start_page(pages):

    print()
    print(
        "Searching for actual story beginning..."
    )

    # --------------------------------------------------------
    # Prefer structural headings outside TOC pages
    # --------------------------------------------------------

    for page in pages:

        if page["is_toc"]:

            print(
                f"Ignoring probable contents page: "
                f"{page['page']}"
            )

            continue

        for block in page["blocks"]:

            if is_primary_story_heading(
                block
            ):

                print()
                print(
                    "Story beginning detected:"
                )

                print(
                    f"Page: {page['page']}"
                )

                print(
                    f"Heading: {block}"
                )

                return page["page"]


    # --------------------------------------------------------
    # FALLBACK
    #
    # If no chapter/part heading exists,
    # find first substantial prose page.
    # --------------------------------------------------------

    print()
    print(
        "No PART/BOOK/CHAPTER heading found."
    )

    print(
        "Using substantial-prose fallback."
    )


    for page in pages:

        if page["is_toc"]:
            continue

        combined = " ".join(
            page["blocks"]
        )

        word_count = len(
            combined.split()
        )

        if word_count >= 150:

            print(
                f"Fallback story start: "
                f"page {page['page']}"
            )

            return page["page"]


    # Absolute fallback
    return 1


# ============================================================
# EXTRACT ACTUAL STORY BLOCKS
# ============================================================

def extract_story_blocks(
    pages,
    story_start_page
):

    story_blocks = []

    current_chapter = "Story Beginning"

    detected_chapters = []


    for page in pages:

        page_number = page["page"]

        # ====================================================
        # Ignore everything BEFORE the detected story start.
        #
        # IMPORTANT:
        # Once the story has started, we DO NOT use the
        # table-of-contents detector anymore because normal
        # story pages can sometimes look like TOC pages.
        # ====================================================

        if page_number < story_start_page:
            continue


        for text in page["blocks"]:

            # ------------------------------------------------
            # Chapter / Part / Book heading
            # ------------------------------------------------

            if is_chapter_heading(text):

                current_chapter = text

                detected_chapters.append({
                    "chapter": text,
                    "page": page_number
                })

                print(
                    f"Detected section: "
                    f"{text} "
                    f"(page {page_number})"
                )

                story_blocks.append({
                    "page": page_number,
                    "chapter": current_chapter,
                    "text": text,
                    "is_heading": True
                })

                continue


            # ------------------------------------------------
            # Normal story content
            # ------------------------------------------------

            story_blocks.append({
                "page": page_number,
                "chapter": current_chapter,
                "text": text,
                "is_heading": False
            })


    print()

    print(
        f"Extracted {len(story_blocks)} "
        f"actual story blocks."
    )

    print(
        f"Detected {len(detected_chapters)} "
        f"story headings."
    )

    return story_blocks


# ============================================================
# SPLIT VERY LARGE BLOCKS
# ============================================================

def split_large_block(block):

    words = block[
        "text"
    ].split()


    if len(words) <= MAX_WORDS:

        return [
            block
        ]


    pieces = []

    start = 0


    while start < len(words):

        end = (
            start
            + TARGET_WORDS
        )

        piece_words = words[
            start:end
        ]

        pieces.append({

            "page":
                block["page"],

            "chapter":
                block["chapter"],

            "text":
                " ".join(
                    piece_words
                ),

            "is_heading":
                block["is_heading"]
        })

        start += TARGET_WORDS


    return pieces


# ============================================================
# NORMALIZE BLOCKS
# ============================================================

def normalize_blocks(blocks):

    normalized = []

    for block in blocks:

        normalized.extend(
            split_large_block(
                block
            )
        )

    return normalized


# ============================================================
# CREATE STORY-AWARE CHUNKS
# ============================================================

def create_story_chunks(
    blocks
):

    chunks = []

    current_blocks = []

    current_words = 0

    current_chapter = None

    chunk_id = 0


    def save_current_chunk():

        nonlocal chunk_id
        nonlocal current_blocks

        if not current_blocks:
            return


        text = "\n\n".join(

            block["text"]

            for block
            in current_blocks
        )


        word_count = len(
            text.split()
        )


        if word_count < 25:
            return


        page_start = (
            current_blocks[0][
                "page"
            ]
        )

        page_end = (
            current_blocks[-1][
                "page"
            ]
        )

        chapter = (
            current_blocks[0][
                "chapter"
            ]
        )


        chunks.append({

            "chunk_id":
                chunk_id,

            # rag.py compatibility
            "page":
                page_start,

            "page_start":
                page_start,

            "page_end":
                page_end,

            "chapter":
                chapter,

            "word_count":
                word_count,

            "text":
                text
        })

        chunk_id += 1


    # ========================================================
    # BUILD CHUNKS
    # ========================================================

    for block in blocks:

        block_words = len(
            block["text"].split()
        )

        block_chapter = (
            block["chapter"]
        )


        # ----------------------------------------------------
        # Chapter changed
        # ----------------------------------------------------

        if (
            current_chapter
            is not None

            and block_chapter
            != current_chapter
        ):

            save_current_chunk()

            current_blocks = []
            current_words = 0


        current_chapter = (
            block_chapter
        )


        # ----------------------------------------------------
        # Existing chunk reached desired size
        # ----------------------------------------------------

        if (
            current_blocks
            and current_words
            >= TARGET_WORDS
        ):

            previous_blocks = (
                current_blocks.copy()
            )

            save_current_chunk()


            # ------------------------------------------------
            # Paragraph overlap
            # ------------------------------------------------

            if (
                OVERLAP_PARAGRAPHS
                > 0
            ):

                current_blocks = (
                    previous_blocks[
                        -OVERLAP_PARAGRAPHS:
                    ]
                )

                current_words = sum(

                    len(
                        item[
                            "text"
                        ].split()
                    )

                    for item
                    in current_blocks
                )

            else:

                current_blocks = []
                current_words = 0


        current_blocks.append(
            block
        )

        current_words += (
            block_words
        )


    # Final chunk
    save_current_chunk()


    print()
    print(
        f"Created {len(chunks)} "
        f"clean story-aware chunks."
    )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks
):

    print()
    print(
        "Loading local embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )


    texts = [

        chunk["text"]

        for chunk
        in chunks
    ]


    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )


    embeddings = model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=True
    )


    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )


    return embeddings


# ============================================================
# SAVE
# ============================================================

def save_data(
    chunks,
    embeddings
):

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )


    print()
    print("Saved:")

    print(
        CHUNKS_FILE
    )

    print(
        EMBEDDINGS_FILE
    )


# ============================================================
# PREVIEW
# ============================================================

def preview_chunks(
    chunks,
    amount=5
):

    print()
    print("=" * 70)
    print("CLEAN STORY CHUNK PREVIEW")
    print("=" * 70)


    for chunk in chunks[
        :amount
    ]:

        print()

        print(
            f"Chunk: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Chapter: "
            f"{chunk['chapter']}"
        )

        print(
            f"Pages: "
            f"{chunk['page_start']} "
            f"→ "
            f"{chunk['page_end']}"
        )

        print(
            f"Words: "
            f"{chunk['word_count']}"
        )

        print()

        print(
            chunk["text"][:700]
        )

        print()

        print(
            "-" * 70
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not BOOK_PATH.exists():

        print(
            f"ERROR: Book not found: "
            f"{BOOK_PATH}"
        )

        return


    # --------------------------------------------------------
    # STEP 1
    # Read PDF
    # --------------------------------------------------------

    pages = read_pdf_pages(
        BOOK_PATH
    )


    # --------------------------------------------------------
    # STEP 2
    # Automatically find real story beginning
    # --------------------------------------------------------

    story_start_page = (
        find_story_start_page(
            pages
        )
    )


    # --------------------------------------------------------
    # STEP 3
    # Remove front matter / TOC
    # --------------------------------------------------------

    blocks = extract_story_blocks(

        pages,

        story_start_page
    )


    # --------------------------------------------------------
    # STEP 4
    # Normalize blocks
    # --------------------------------------------------------

    blocks = normalize_blocks(
        blocks
    )


    # --------------------------------------------------------
    # STEP 5
    # Story-aware chunks
    # --------------------------------------------------------

    chunks = create_story_chunks(
        blocks
    )


    if not chunks:

        print(
            "ERROR: No story chunks created."
        )

        return


    # --------------------------------------------------------
    # STEP 6
    # Preview CLEAN chunks
    # --------------------------------------------------------

    preview_chunks(
        chunks
    )


    # --------------------------------------------------------
    # STEP 7
    # Local embeddings
    # --------------------------------------------------------

    embeddings = (
        create_embeddings(
            chunks
        )
    )


    # --------------------------------------------------------
    # STEP 8
    # Save
    # --------------------------------------------------------

    save_data(
        chunks,
        embeddings
    )


    print()
    print("=" * 70)

    print(
        "Clean StoryWorld memory "
        "created successfully."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()