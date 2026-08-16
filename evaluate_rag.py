import json
import re
import time
from pathlib import Path

import rag


# ============================================================
# SETTINGS
# ============================================================

QUESTIONS_FILE = Path(
    "eval_questions.json"
)

RESULTS_FILE = Path(
    "data/evaluation_results.json"
)

SUMMARY_FILE = Path(
    "data/evaluation_summary.json"
)


# Delay between Gemini requests.
REQUEST_DELAY_SECONDS = 4


UNKNOWN_RESPONSE = (
    "The available story evidence does not establish that."
)


# ============================================================
# STORY-ONLY SYSTEM INSTRUCTION
# ============================================================

STORY_ONLY_SYSTEM = """
You are StoryWorld.

You answer questions ONLY using the supplied direct
story excerpts.

RULES:

1. The supplied story excerpts are your ONLY factual
   knowledge.

2. Do NOT use outside knowledge about the book,
   Sherlock Holmes, the author, adaptations, history,
   or anything else.

3. Never invent facts.

4. Only make claims supported by the supplied passages.

5. You may make a reasonable inference only when the
   supplied passages clearly support it.

6. If evidence is insufficient, say exactly:

   "The available story evidence does not establish that."

7. Cite supporting pages using:

   [Page X]

   or

   [Pages X-Y]

8. Keep the answer clear and concise.
"""


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    if not QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"{QUESTIONS_FILE} was not found."
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    return " ".join(
        text.lower()
        .strip()
        .split()
    )


# ============================================================
# CONCEPT COVERAGE
# ============================================================

def concept_coverage(
    text,
    expected_groups
):

    if not expected_groups:

        return 1.0

    normalized = normalize(
        text
    )

    matched = 0


    for group in expected_groups:

        group_found = False


        for alternative in group:

            if normalize(
                alternative
            ) in normalized:

                group_found = True

                break


        if group_found:

            matched += 1


    return (
        matched
        / len(expected_groups)
    )


# ============================================================
# CITATION CHECK
# ============================================================

def has_page_citation(
    answer
):

    return bool(
        re.search(
            r"\[Pages?\s+[0-9]+",
            answer,
            re.IGNORECASE
        )
    )


# ============================================================
# UNKNOWN ANSWER CHECK
# ============================================================

def correctly_refused(
    answer
):

    return (
        normalize(
            UNKNOWN_RESPONSE
        )
        in normalize(
            answer
        )
    )


# ============================================================
# EXTRACT RETRY WAIT FROM GEMINI ERROR
# ============================================================

def get_retry_seconds(
    error_text
):

    match = re.search(
        r"retry in ([0-9.]+)s",
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
# SAFE GEMINI REQUEST
# ============================================================

def call_gemini(
    system_instruction,
    prompt
):

    for attempt in range(
        1,
        4
    ):

        try:

            interaction = (
                rag.client
                .interactions
                .create(

                    model=
                        rag.LLM_MODEL,

                    system_instruction=
                        system_instruction,

                    input=
                        prompt,

                    store=False
                )
            )


            return (
                interaction.output_text
            )


        except Exception as error:

            error_text = str(
                error
            )


            print()

            print(
                f"Gemini attempt "
                f"{attempt} failed:"
            )

            print(
                error_text
            )


            if attempt == 3:

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
                f"Waiting "
                f"{wait_seconds:.0f}s..."
            )


            time.sleep(
                wait_seconds
            )


# ============================================================
# STORY-ONLY ANSWER
# ============================================================

def generate_story_only(
    question,
    story_results
):

    story_context = (
        rag.build_story_context(
            story_results
        )
    )


    prompt = f"""
USER QUESTION:

{question}


============================================================
DIRECT STORY EVIDENCE
============================================================

{story_context}


Answer the question using ONLY the supplied
direct story passages.
"""


    return call_gemini(

        STORY_ONLY_SYSTEM,

        prompt
    )


# ============================================================
# HYBRID ANSWER
# ============================================================

def generate_hybrid(
    question,
    story_results,
    world_results
):

    story_context = (
        rag.build_story_context(
            story_results
        )
    )


    world_context = (
        rag.build_world_context(
            world_results
        )
    )


    prompt = f"""
USER QUESTION:

{question}


============================================================
DIRECT STORY MEMORY
============================================================

{story_context}


============================================================
STRUCTURED WORLD MEMORY
============================================================

{world_context}


============================================================
TASK
============================================================

Answer the question using ONLY the supplied evidence.

Story Memory is the primary source of truth.

Use World Memory to help understand characters,
relationships, facts, events, and locations.

If evidence is insufficient, say exactly:

"The available story evidence does not establish that."
"""


    return call_gemini(

        rag.SYSTEM_INSTRUCTION,

        prompt
    )


# ============================================================
# BUILD EVIDENCE TEXT
# ============================================================

def story_evidence_text(
    results
):

    return "\n".join(

        result[
            "text"
        ]

        for result
        in results
    )


def world_evidence_text(
    results
):

    return "\n".join(

        result[
            "text"
        ]

        for result
        in results
    )


# ============================================================
# SCORE GENERATED ANSWER
# ============================================================

def score_answer(
    answer,
    question_data
):

    unanswerable = (
        question_data[
            "unanswerable"
        ]
    )


    # --------------------------------------------------------
    # Unknown / unsupported question
    # --------------------------------------------------------

    if unanswerable:

        refusal = correctly_refused(
            answer
        )


        return {

            "concept_coverage":
                1.0
                if refusal
                else 0.0,

            "citation":
                False,

            "correct_refusal":
                refusal,

            "quality_score":
                1.0
                if refusal
                else 0.0
        }


    # --------------------------------------------------------
    # Answerable question
    # --------------------------------------------------------

    coverage = concept_coverage(

        answer,

        question_data[
            "expected_groups"
        ]
    )


    citation = has_page_citation(
        answer
    )


    # 80% important concepts
    # 20% citation behavior

    quality_score = (
        coverage * 0.8
        +
        (
            0.2
            if citation
            else 0.0
        )
    )


    return {

        "concept_coverage":
            round(
                coverage,
                4
            ),

        "citation":
            citation,

        "correct_refusal":
            False,

        "quality_score":
            round(
                quality_score,
                4
            )
    }


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    question_data
):

    question = (
        question_data[
            "question"
        ]
    )


    print()
    print("=" * 70)

    print(
        f"QUESTION "
        f"{question_data['id']}"
    )

    print("=" * 70)

    print(
        question
    )


    # --------------------------------------------------------
    # Embed once
    # --------------------------------------------------------

    question_embedding = (
        rag.embed_question(
            question
        )
    )


    # --------------------------------------------------------
    # Retrieve Story Memory
    # --------------------------------------------------------

    story_results = (
        rag.search_story(
            question_embedding
        )
    )


    # --------------------------------------------------------
    # Retrieve World Memory
    # --------------------------------------------------------

    world_results = (
        rag.search_world(
            question_embedding
        )
    )


    # --------------------------------------------------------
    # Retrieval coverage
    # --------------------------------------------------------

    story_text = (
        story_evidence_text(
            story_results
        )
    )


    world_text = (
        world_evidence_text(
            world_results
        )
    )


    expected_groups = (
        question_data[
            "expected_groups"
        ]
    )


    story_retrieval_coverage = (
        concept_coverage(
            story_text,
            expected_groups
        )
    )


    hybrid_retrieval_coverage = (
        concept_coverage(

            story_text
            + "\n"
            + world_text,

            expected_groups
        )
    )


    # --------------------------------------------------------
    # STORY-ONLY generation
    # --------------------------------------------------------

    print()
    print(
        "Generating Story-only answer..."
    )


    story_answer = (
        generate_story_only(

            question,

            story_results
        )
    )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


    # --------------------------------------------------------
    # HYBRID generation
    # --------------------------------------------------------

    print(
        "Generating Hybrid answer..."
    )


    hybrid_answer = (
        generate_hybrid(

            question,

            story_results,

            world_results
        )
    )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


    # --------------------------------------------------------
    # Score answers
    # --------------------------------------------------------

    story_score = (
        score_answer(

            story_answer,

            question_data
        )
    )


    hybrid_score = (
        score_answer(

            hybrid_answer,

            question_data
        )
    )


    print()
    print(
        "Story-only:"
    )

    print(
        story_answer
    )

    print()

    print(
        "Hybrid:"
    )

    print(
        hybrid_answer
    )


    print()

    print(
        f"Story score : "
        f"{story_score['quality_score']:.2f}"
    )

    print(
        f"Hybrid score: "
        f"{hybrid_score['quality_score']:.2f}"
    )


    # --------------------------------------------------------
    # Determine winner
    # --------------------------------------------------------

    if (
        hybrid_score[
            "quality_score"
        ]
        >
        story_score[
            "quality_score"
        ]
    ):

        winner = "hybrid"


    elif (
        story_score[
            "quality_score"
        ]
        >
        hybrid_score[
            "quality_score"
        ]
    ):

        winner = "story_only"


    else:

        winner = "tie"


    return {

        "id":
            question_data[
                "id"
            ],

        "question":
            question,

        "category":
            question_data[
                "category"
            ],

        "unanswerable":
            question_data[
                "unanswerable"
            ],

        "retrieval": {

            "story_only_coverage":
                round(
                    story_retrieval_coverage,
                    4
                ),

            "hybrid_coverage":
                round(
                    hybrid_retrieval_coverage,
                    4
                )
        },

        "story_only": {

            "answer":
                story_answer,

            "scores":
                story_score
        },

        "hybrid": {

            "answer":
                hybrid_answer,

            "scores":
                hybrid_score
        },

        "winner":
            winner
    }


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    results
):

    total = len(
        results
    )


    story_total = sum(

        item[
            "story_only"
        ][
            "scores"
        ][
            "quality_score"
        ]

        for item
        in results
    )


    hybrid_total = sum(

        item[
            "hybrid"
        ][
            "scores"
        ][
            "quality_score"
        ]

        for item
        in results
    )


    story_retrieval = sum(

        item[
            "retrieval"
        ][
            "story_only_coverage"
        ]

        for item
        in results
    )


    hybrid_retrieval = sum(

        item[
            "retrieval"
        ][
            "hybrid_coverage"
        ]

        for item
        in results
    )


    hybrid_wins = sum(

        1

        for item
        in results

        if item[
            "winner"
        ] == "hybrid"
    )


    story_wins = sum(

        1

        for item
        in results

        if item[
            "winner"
        ] == "story_only"
    )


    ties = sum(

        1

        for item
        in results

        if item[
            "winner"
        ] == "tie"
    )


    return {

        "questions":
            total,

        "story_only_average_answer_score":
            round(
                story_total / total,
                4
            ),

        "hybrid_average_answer_score":
            round(
                hybrid_total / total,
                4
            ),

        "story_only_average_retrieval_coverage":
            round(
                story_retrieval / total,
                4
            ),

        "hybrid_average_retrieval_coverage":
            round(
                hybrid_retrieval / total,
                4
            ),

        "hybrid_wins":
            hybrid_wins,

        "story_only_wins":
            story_wins,

        "ties":
            ties
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STORYWORLD RAG EVALUATION"
    )
    print("=" * 70)

    print()

    print(
        "Comparing:"
    )

    print(
        "A. Story-only RAG"
    )

    print(
        "B. Story + World Memory RAG"
    )


    questions = (
        load_questions()
    )


    print()

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print(
        f"Expected Gemini calls: "
        f"{len(questions) * 2}"
    )


    results = []


    for question in questions:

        result = (
            evaluate_question(
                question
            )
        )

        results.append(
            result
        )


        # Save continuously
        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                ensure_ascii=False,
                indent=2
            )


    summary = build_summary(
        results
    )


    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 70)

    print(
        "FINAL EVALUATION RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Story-only answer score : "
        f"{summary['story_only_average_answer_score']:.2%}"
    )

    print(
        f"Hybrid answer score     : "
        f"{summary['hybrid_average_answer_score']:.2%}"
    )

    print()

    print(
        f"Story retrieval coverage: "
        f"{summary['story_only_average_retrieval_coverage']:.2%}"
    )

    print(
        f"Hybrid retrieval coverage: "
        f"{summary['hybrid_average_retrieval_coverage']:.2%}"
    )

    print()

    print(
        f"Hybrid wins    : "
        f"{summary['hybrid_wins']}"
    )

    print(
        f"Story-only wins: "
        f"{summary['story_only_wins']}"
    )

    print(
        f"Ties           : "
        f"{summary['ties']}"
    )

    print()

    print(
        "Detailed results:"
    )

    print(
        RESULTS_FILE
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":

    main()