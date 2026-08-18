"use client";

import Link from "next/link";
import { FormEvent, useEffect, useRef, useState } from "react";

type StoryInfo = {
  title: string;
  story_chunks: number;
  world_items: number;
  embedding_model: string;
  generation_model: string;
  status: string;
};

type StorySource = {
  chunk_id: number | null;
  chapter: string;
  page_start: number | null;
  page_end: number | null;
  excerpt: string;
};

type Character = {
  name: string;
  description: string;
  sources: {
    chunk_id?: number;
    chapter?: string;
    page_start?: number;
    page_end?: number;
  }[];
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: StorySource[];
};

const suggestions = [
  "Who is Helen Stoner?",
  "Why was Dr. Roylott against Helen's marriage?",
  "What happened to Julia Stoner?",
  "Why was Holmes suspicious of the bell-rope?",
];

export default function Home() {
  const [story, setStory] = useState<StoryInfo | null>(null);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [backendOnline, setBackendOnline] = useState(false);

  const [view, setView] = useState<"chat" | "characters">("chat");

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome to StoryWorld. Ask me anything about the story. My answers are grounded in the book and supported by retrieved passages.",
    },
  ]);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [healthResponse, storyResponse, characterResponse] =
          await Promise.all([
            fetch("/api/storyworld/health"),
            fetch("/api/storyworld/story"),
            fetch("/api/storyworld/characters"),
          ]);

        setBackendOnline(healthResponse.ok);

        if (storyResponse.ok) {
          setStory(await storyResponse.json());
        }

        if (characterResponse.ok) {
          setCharacters(await characterResponse.json());
        }
      } catch {
        setBackendOnline(false);
      }
    }

    loadData();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  async function askQuestion(text?: string) {
    const value = (text ?? question).trim();

    if (!value || loading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: value,
    };

    setMessages((current) => [...current, userMessage]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("/api/storyworld/chat", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          question: value,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Request failed.");
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? [],
      };

      setMessages((current) => [
        ...current,
        assistantMessage,
      ]);
    } catch {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "I couldn't reach the StoryWorld reasoning engine. Check that the FastAPI backend is running and try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    askQuestion();
  }

  return (
    <main className="min-h-screen bg-[#090b10] text-zinc-100">
      <div className="mx-auto grid min-h-screen max-w-[1700px] lg:grid-cols-[300px_1fr]">

        {/* SIDEBAR */}
        <aside className="border-b border-white/10 bg-[#0d1017] p-5 lg:border-b-0 lg:border-r lg:p-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-300 font-serif text-xl font-bold text-zinc-950 shadow-lg shadow-amber-300/10">
              S
            </div>

            <div>
              <h1 className="font-serif text-xl font-semibold">
                StoryWorld
              </h1>

              <p className="text-xs text-zinc-500">
                Story intelligence
              </p>
            </div>
          </div>

          <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-amber-300">
              Current story
            </p>

            <h2 className="mt-3 font-serif text-lg leading-snug text-white">
              {story?.title ?? "Loading story..."}
            </h2>

            <div className="mt-5 grid grid-cols-2 gap-3">
              <Stat
                value={story?.story_chunks ?? "—"}
                label="Story chunks"
              />

              <Stat
                value={story?.world_items ?? "—"}
                label="World facts"
              />
            </div>
          </div>

          <nav className="mt-7 space-y-2">
            <NavButton
              active={view === "chat"}
              onClick={() => setView("chat")}
              icon="✦"
              label="Story Chat"
            />

            <NavButton
              active={view === "characters"}
              onClick={() => setView("characters")}
              icon="◎"
              label="Characters"
            />

            <Link
              href="/relationships"
              className="flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left text-sm text-zinc-400 transition hover:bg-white/[0.05] hover:text-white"
            >
              <span>⌘</span>
              <span className="font-medium">
                Relationship Map
              </span>
            </Link>
          </nav>

          <div className="mt-8 border-t border-white/10 pt-5">
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline
                    ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.7)]"
                    : "bg-red-400"
                }`}
              />

              {backendOnline
                ? "StoryWorld ready"
                : "Backend offline"}
            </div>

            <p className="mt-2 text-xs leading-relaxed text-zinc-600">
              Story retrieval and world retrieval run locally.
              Gemini generates the final grounded answer.
            </p>
          </div>
        </aside>

        {/* MAIN */}
        <section className="flex min-h-0 flex-col">

          {/* HEADER */}
          <header className="flex min-h-20 items-center justify-between border-b border-white/10 px-5 py-4 md:px-8">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">
                {view === "chat"
                  ? "Story conversation"
                  : "World explorer"}
              </p>

              <h2 className="mt-1 font-serif text-xl text-zinc-100">
                {view === "chat"
                  ? "Ask the story"
                  : "Characters"}
              </h2>
            </div>

            <div className="hidden rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs text-zinc-400 md:block">
              Hybrid Story + World RAG
            </div>
          </header>

          {view === "chat" ? (
            <ChatView
              messages={messages}
              loading={loading}
              question={question}
              setQuestion={setQuestion}
              handleSubmit={handleSubmit}
              askQuestion={askQuestion}
              bottomRef={bottomRef}
            />
          ) : (
            <CharacterView characters={characters} />
          )}
        </section>
      </div>
    </main>
  );
}

function Stat({
  value,
  label,
}: {
  value: string | number;
  label: string;
}) {
  return (
    <div className="rounded-xl bg-black/20 p-3">
      <div className="text-lg font-semibold text-white">
        {value}
      </div>

      <div className="mt-1 text-[11px] text-zinc-500">
        {label}
      </div>
    </div>
  );
}

function NavButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: string;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left text-sm transition ${
        active
          ? "bg-amber-300 text-zinc-950"
          : "text-zinc-400 hover:bg-white/[0.05] hover:text-white"
      }`}
    >
      <span>{icon}</span>
      <span className="font-medium">{label}</span>
    </button>
  );
}

function ChatView({
  messages,
  loading,
  question,
  setQuestion,
  handleSubmit,
  askQuestion,
  bottomRef,
}: {
  messages: ChatMessage[];
  loading: boolean;
  question: string;
  setQuestion: (value: string) => void;
  handleSubmit: (event: FormEvent<HTMLFormElement>) => void;
  askQuestion: (text?: string) => void;
  bottomRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 overflow-y-auto px-5 py-7 md:px-8">
        <div className="mx-auto max-w-4xl">

          {messages.length === 1 && (
            <div className="mb-10 rounded-3xl border border-white/10 bg-gradient-to-br from-white/[0.05] to-transparent p-6 md:p-8">
              <p className="text-xs font-medium uppercase tracking-[0.2em] text-amber-300">
                Explore the mystery
              </p>

              <h3 className="mt-3 max-w-xl font-serif text-3xl leading-tight text-white md:text-4xl">
                Enter the world of the story.
              </h3>

              <p className="mt-4 max-w-2xl leading-7 text-zinc-400">
                Ask about characters, motives, relationships,
                clues, events, or anything established by the
                book.
              </p>

              <div className="mt-6 grid gap-3 md:grid-cols-2">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => askQuestion(suggestion)}
                    className="rounded-2xl border border-white/10 bg-black/20 p-4 text-left text-sm leading-6 text-zinc-300 transition hover:border-amber-300/40 hover:bg-amber-300/[0.05] hover:text-white"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-7">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message}
              />
            ))}

            {loading && <Thinking />}

            <div ref={bottomRef} />
          </div>
        </div>
      </div>

      {/* INPUT */}
      <div className="border-t border-white/10 bg-[#090b10]/95 px-5 py-5 md:px-8">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-4xl items-end gap-3 rounded-2xl border border-white/10 bg-[#11151d] p-2 shadow-2xl shadow-black/20 focus-within:border-amber-300/30"
        >
          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey
              ) {
                event.preventDefault();

                if (question.trim() && !loading) {
                  askQuestion();
                }
              }
            }}
            rows={1}
            placeholder="Ask something about the story..."
            className="max-h-36 min-h-12 flex-1 resize-none bg-transparent px-4 py-3 text-sm text-white outline-none placeholder:text-zinc-600"
          />

          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="flex h-12 items-center justify-center rounded-xl bg-amber-300 px-5 text-sm font-semibold text-zinc-950 transition hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Ask
          </button>
        </form>

        <p className="mx-auto mt-2 max-w-4xl text-center text-[11px] text-zinc-600">
          Answers are generated only from retrieved story evidence.
        </p>
      </div>
    </div>
  );
}

function Message({
  message,
}: {
  message: ChatMessage;
}) {
  const user = message.role === "user";

  return (
    <div
      className={`flex ${
        user ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-3xl ${
          user
            ? "rounded-2xl bg-amber-300 px-5 py-3 text-zinc-950"
            : "w-full"
        }`}
      >
        {!user && (
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-300 text-xs font-bold text-zinc-950">
              S
            </div>

            <span className="text-xs font-medium text-zinc-400">
              StoryWorld
            </span>
          </div>
        )}

        <div
          className={`whitespace-pre-wrap text-sm leading-7 ${
            user
              ? "font-medium"
              : "text-zinc-200"
          }`}
        >
          {message.content}
        </div>

        {!user &&
          message.sources &&
          message.sources.length > 0 && (
            <SourceList sources={message.sources} />
          )}
      </div>
    </div>
  );
}

function SourceList({
  sources,
}: {
  sources: StorySource[];
}) {
  return (
    <div className="mt-5 border-t border-white/10 pt-4">
      <p className="mb-3 text-xs font-medium uppercase tracking-[0.15em] text-zinc-500">
        Retrieved sources
      </p>

      <div className="space-y-2">
        {sources.slice(0, 3).map((source) => (
          <details
            key={`${source.chunk_id}-${source.page_start}`}
            className="group rounded-xl border border-white/10 bg-white/[0.025]"
          >
            <summary className="cursor-pointer list-none px-4 py-3 text-sm text-zinc-400 transition hover:text-white">
              <div className="flex items-center justify-between gap-4">
                <span>
                  Pages {source.page_start}
                  {source.page_end !== source.page_start
                    ? `–${source.page_end}`
                    : ""}
                </span>

                <span className="text-zinc-600 group-open:rotate-45">
                  +
                </span>
              </div>
            </summary>

            <div className="border-t border-white/10 px-4 py-4">
              <p className="text-xs font-medium text-amber-300">
                {source.chapter}
              </p>

              <p className="mt-2 text-sm leading-6 text-zinc-400">
                {source.excerpt}
              </p>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-300 text-xs font-bold text-zinc-950">
          S
        </div>

        <span className="text-xs text-zinc-500">
          Searching the story...
        </span>
      </div>

      <div className="flex gap-1.5">
        {[0, 1, 2].map((item) => (
          <div
            key={item}
            className="h-2 w-2 animate-pulse rounded-full bg-zinc-600"
          />
        ))}
      </div>
    </div>
  );
}

function CharacterView({
  characters,
}: {
  characters: Character[];
}) {
  return (
    <div className="flex-1 overflow-y-auto p-5 md:p-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-8">
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-amber-300">
            World Memory
          </p>

          <h3 className="mt-2 font-serif text-3xl text-white">
            Characters in this story
          </h3>

          <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-500">
            Character identities are consolidated from story
            passages and semantic entity resolution.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {characters.map((character) => (
            <article
              key={character.name}
              className="rounded-2xl border border-white/10 bg-white/[0.025] p-5 transition hover:border-amber-300/20 hover:bg-white/[0.04]"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-300/10 font-serif text-lg text-amber-300">
                {character.name.charAt(0)}
              </div>

              <h4 className="mt-4 font-serif text-xl text-white">
                {character.name}
              </h4>

              <p className="mt-3 text-sm leading-6 text-zinc-400">
                {character.description ||
                  "No description available."}
              </p>

              <p className="mt-4 text-xs text-zinc-600">
                {character.sources.length} supporting passage
                {character.sources.length === 1 ? "" : "s"}
              </p>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}