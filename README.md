# StoryWorld — Hybrid RAG for Fictional Worlds

StoryWorld is a full-stack **Retrieval-Augmented Generation (RAG)** system that answers questions about a fictional world using evidence from the source text.

Instead of relying on a single vector index, StoryWorld combines two complementary memory layers:

- **Story Memory** — direct passages retrieved from the source text
- **World Memory** — structured knowledge about characters, relationships, locations, events, and facts derived from the story

This architecture helps connect information distributed across a narrative while keeping answers grounded in traceable evidence.

## Architecture

```text
User Question
      |
      v
SentenceTransformer
      |
      +-------------------+
      |                   |
      v                   v
 Story Memory         World Memory
 Direct passages      Structured knowledge
      |                   |
      +---------+---------+
                |
                v
         Evidence Context
                |
                v
         Gemini Generation
                |
                v
     Grounded Answer + Sources
```

## Key Features

- Hybrid retrieval across direct passages and structured world knowledge
- Local semantic embeddings using `all-MiniLM-L6-v2`
- Evidence-grounded generation with Gemini
- Character and relationship extraction
- Entity resolution across structured world knowledge
- Chapter and page-level source metadata
- RAG evaluation pipeline
- FastAPI backend with typed request and response models
- Next.js frontend
- Debug mode for inspecting retrieved evidence
- Explicit fallback when available evidence cannot establish an answer

## Tech Stack

**AI & Retrieval:** Python · Sentence Transformers · NumPy · Gemini  
**Backend:** FastAPI · Pydantic · Uvicorn  
**Frontend:** Next.js · React · TypeScript · Tailwind CSS

## Repository Highlights

```text
api.py                  FastAPI application and endpoints
rag.py                  Hybrid retrieval and generation pipeline
build_world_index.py    Builds the structured World Memory index
entity_resolver.py      Resolves entities across extracted knowledge
evaluate_rag.py         RAG evaluation pipeline
eval_questions.json     Evaluation question set
frontend/               Next.js user interface
data/                   Story and World Memory artifacts
```

## How It Works

1. The user question is embedded with Sentence Transformers.
2. The embedding searches Story Memory for relevant passages.
3. The same embedding searches World Memory for related structured knowledge.
4. The highest-ranking evidence from both memories is assembled into context.
5. Gemini generates an answer under strict grounding instructions.
6. The API returns the answer together with supporting story passages.
7. When evidence is insufficient, the system explicitly states that the available story evidence does not establish the answer.

## API

- `GET /api/health` — service and model status
- `GET /api/story` — story and model information
- `GET /api/characters` — extracted characters
- `GET /api/relationships` — resolved relationships
- `POST /api/chat` — grounded story question answering

Interactive API documentation is available at `/docs` while the backend is running.

## Setup

### Backend

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Configure the credentials required by the Google Gen AI SDK before starting generation.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on port 3000 by default and communicates with the FastAPI backend on port 8000.

## Grounding Strategy

StoryWorld treats retrieved text as **evidence, not instructions**. Direct Story Memory remains the primary source of truth, while World Memory helps connect entities, relationships, events, and locations.

If structured memory conflicts with direct story evidence, direct story evidence takes priority.

## Why This Project

Basic semantic RAG can retrieve relevant passages but may struggle with relationships distributed across a long narrative. StoryWorld explores a hybrid approach that combines semantic retrieval with structured world knowledge while preserving traceability to the original source.

---

**Focus:** RAG · LLM Applications · Knowledge Extraction · Semantic Retrieval · Full-Stack AI
