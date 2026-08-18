"use client";

import Link from "next/link";

import {
  useEffect,
  useMemo,
  useState,
} from "react";


type Source = {
  chunk_id?: number;
  chapter?: string;
  page_start?: number;
  page_end?: number;
};


type Character = {
  name: string;
  description: string;
  sources: Source[];
};


type Relationship = {
  subject: string;
  relation: string;
  object: string;
  explanation: string;
  sources: Source[];
};


type GraphNode = {
  name: string;
  x: number;
  y: number;
};


function prettyRelation(
  relation: string
) {
  return relation
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}


function shortName(
  name: string
) {
  const words = name.split(" ");

  if (words.length <= 2) {
    return name;
  }

  return `${words[0]} ${words[
    words.length - 1
  ]}`;
}


export default function RelationshipsPage() {

  const [
    characters,
    setCharacters
  ] = useState<Character[]>([]);

  const [
    relationships,
    setRelationships
  ] = useState<Relationship[]>([]);

  const [
    loading,
    setLoading
  ] = useState(true);

  const [
    selected,
    setSelected
  ] = useState<string | null>(
    null
  );


  // ========================================================
  // LOAD DATA
  // ========================================================

  useEffect(() => {

    async function loadWorld() {

      try {

        const [
          characterResponse,
          relationshipResponse
        ] = await Promise.all([

          fetch(
            "/api/storyworld/characters"
          ),

          fetch(
            "/api/storyworld/relationships"
          )
        ]);


        if (
          !characterResponse.ok ||
          !relationshipResponse.ok
        ) {

          throw new Error(
            "Could not load World Memory."
          );
        }


        const characterData =
          await characterResponse.json();

        const relationshipData =
          await relationshipResponse.json();


        setCharacters(
          characterData
        );

        setRelationships(
          relationshipData
        );

      } catch (error) {

        console.error(
          error
        );

      } finally {

        setLoading(false);
      }
    }


    loadWorld();

  }, []);


  // ========================================================
  // CHARACTER NAMES
  // ========================================================

  const characterNames =
    useMemo(() => {

      return new Set(
        characters.map(
          (character) =>
            character.name
        )
      );

    }, [characters]);


  // ========================================================
  // ONLY CHARACTER ↔ CHARACTER RELATIONSHIPS
  // ========================================================

  const characterRelationships =
    useMemo(() => {

      return relationships.filter(
        (relationship) =>

          characterNames.has(
            relationship.subject
          )

          &&

          characterNames.has(
            relationship.object
          )

          &&

          relationship.subject
            !==
          relationship.object
      );

    }, [
      relationships,
      characterNames
    ]);


  // ========================================================
  // FIND MOST CONNECTED CHARACTERS
  // ========================================================

  const topCharacterNames =
    useMemo(() => {

      const counts =
        new Map<
          string,
          number
        >();


      characterRelationships.forEach(
        (relationship) => {

          counts.set(
            relationship.subject,

            (
              counts.get(
                relationship.subject
              )
              ?? 0
            ) + 1
          );


          counts.set(
            relationship.object,

            (
              counts.get(
                relationship.object
              )
              ?? 0
            ) + 1
          );
        }
      );


      return [
        ...counts.entries()
      ]

        .sort(
          (a, b) =>
            b[1] - a[1]
        )

        .slice(
          0,
          10
        )

        .map(
          ([name]) =>
            name
        );

    }, [
      characterRelationships
    ]);


  // ========================================================
  // SET DEFAULT SELECTED CHARACTER
  // ========================================================

  useEffect(() => {

    if (
      !selected
      &&
      topCharacterNames.length
      > 0
    ) {

      setSelected(
        topCharacterNames[0]
      );
    }

  }, [
    topCharacterNames,
    selected
  ]);


  // ========================================================
  // GRAPH NODES
  // ========================================================

  const nodes =
    useMemo(() => {

      const centerX = 450;

      const centerY = 300;

      const radius = 220;


      return topCharacterNames.map(
        (name, index) => {

          const angle =
            (
              (Math.PI * 2)
              /
              topCharacterNames.length
            )
            * index
            -
            Math.PI / 2;


          return {

            name,

            x:
              centerX
              +
              Math.cos(
                angle
              )
              * radius,

            y:
              centerY
              +
              Math.sin(
                angle
              )
              * radius
          };
        }
      );

    }, [
      topCharacterNames
    ]);


  const nodeMap =
    useMemo(() => {

      return new Map(
        nodes.map(
          (node) => [
            node.name,
            node
          ]
        )
      );

    }, [nodes]);


  // ========================================================
  // GRAPH EDGES
  // ========================================================

  const graphRelationships =
    useMemo(() => {

      const nodeNames =
        new Set(
          topCharacterNames
        );

      return characterRelationships.filter(
        (relationship) =>

          nodeNames.has(
            relationship.subject
          )

          &&

          nodeNames.has(
            relationship.object
          )
      );

    }, [
      characterRelationships,
      topCharacterNames
    ]);


  // ========================================================
  // SELECTED CHARACTER RELATIONSHIPS
  // ========================================================

  const selectedRelationships =
    useMemo(() => {

      if (!selected) {
        return [];
      }


      return characterRelationships.filter(
        (relationship) =>

          relationship.subject
            === selected

          ||

          relationship.object
            === selected
      );

    }, [
      characterRelationships,
      selected
    ]);


  const selectedCharacter =
    characters.find(
      (character) =>
        character.name
        === selected
    );


  // ========================================================
  // LOADING
  // ========================================================

  if (loading) {

    return (
      <main className="flex min-h-screen items-center justify-center bg-[#090b10] text-zinc-400">

        Loading StoryWorld...

      </main>
    );
  }


  return (

    <main className="min-h-screen bg-[#090b10] text-white">

      {/* HEADER */}

      <header className="border-b border-white/10 bg-[#0d1017]">

        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-5">

          <div>

            <a
              href="/"
              className="text-xs text-zinc-500 transition hover:text-amber-300"
            >
              ← Back to Story Chat
            </a>


            <h1 className="mt-2 font-serif text-2xl">

              Relationship Map

            </h1>

          </div>


          <div className="hidden rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-xs text-zinc-400 sm:block">

            StoryWorld Memory

          </div>

        </div>

      </header>


      <div className="mx-auto max-w-[1500px] px-5 py-8 md:px-8">


        {/* INTRO */}

        <div className="mb-8">

          <p className="text-xs font-medium uppercase tracking-[0.18em] text-amber-300">

            Character intelligence

          </p>


          <h2 className="mt-2 max-w-3xl font-serif text-3xl leading-tight md:text-4xl">

            Explore how the people in the story connect.

          </h2>


          <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-500">

            The map is generated from relationships extracted from
            the story's structured World Memory.

          </p>

        </div>


        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.7fr)_minmax(340px,.8fr)]">


          {/* GRAPH */}

          <section className="overflow-hidden rounded-3xl border border-white/10 bg-[#0d1017]">

            <div className="border-b border-white/10 px-5 py-4">

              <div className="flex items-center justify-between">

                <div>

                  <p className="text-sm font-medium">

                    Character Network

                  </p>

                  <p className="mt-1 text-xs text-zinc-600">

                    Click a character to explore their relationships.

                  </p>

                </div>


                <div className="rounded-lg bg-white/[0.04] px-3 py-2 text-xs text-zinc-500">

                  {graphRelationships.length} connections

                </div>

              </div>

            </div>


            <div className="overflow-x-auto">

              <svg
                viewBox="0 0 900 600"
                className="min-h-[520px] min-w-[760px] w-full"
              >

                {/* EDGES */}

                {graphRelationships.map(
                  (
                    relationship,
                    index
                  ) => {

                    const start =
                      nodeMap.get(
                        relationship.subject
                      );

                    const end =
                      nodeMap.get(
                        relationship.object
                      );


                    if (
                      !start
                      ||
                      !end
                    ) {

                      return null;
                    }


                    const highlighted =
                      selected
                      === relationship.subject

                      ||

                      selected
                      === relationship.object;


                    return (

                      <line
                        key={`${relationship.subject}-${relationship.object}-${relationship.relation}-${index}`}

                        x1={start.x}
                        y1={start.y}

                        x2={end.x}
                        y2={end.y}

                        stroke={
                          highlighted
                            ? "#fcd34d"
                            : "#343944"
                        }

                        strokeWidth={
                          highlighted
                            ? 2.5
                            : 1
                        }

                        opacity={
                          selected
                            ? highlighted
                              ? 0.9
                              : 0.14
                            : 0.4
                        }
                      />

                    );
                  }
                )}


                {/* NODES */}

                {nodes.map(
                  (node) => {

                    const active =
                      selected
                      === node.name;


                    return (

                      <g
                        key={node.name}

                        onClick={() =>
                          setSelected(
                            node.name
                          )
                        }

                        className="cursor-pointer"
                      >

                        <circle
                          cx={node.x}
                          cy={node.y}

                          r={
                            active
                              ? 48
                              : 40
                          }

                          fill={
                            active
                              ? "#fcd34d"
                              : "#161b24"
                          }

                          stroke={
                            active
                              ? "#fcd34d"
                              : "#444a57"
                          }

                          strokeWidth="2"
                        />


                        <text
                          x={node.x}
                          y={node.y - 4}

                          textAnchor="middle"

                          fill={
                            active
                              ? "#090b10"
                              : "#ffffff"
                          }

                          fontSize="13"

                          fontWeight="600"
                        >

                          {
                            shortName(
                              node.name
                            )
                          }

                        </text>


                        <text
                          x={node.x}
                          y={node.y + 16}

                          textAnchor="middle"

                          fill={
                            active
                              ? "#3f3f46"
                              : "#71717a"
                          }

                          fontSize="10"
                        >

                          {
                            characterRelationships.filter(
                              (relationship) =>

                                relationship.subject
                                === node.name

                                ||

                                relationship.object
                                === node.name
                            ).length
                          } connections

                        </text>

                      </g>

                    );
                  }
                )}

              </svg>

            </div>

          </section>


          {/* CHARACTER DETAILS */}

          <aside className="rounded-3xl border border-white/10 bg-[#0d1017] p-6">

            {
              selectedCharacter
              ? (
                <>

                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-300 font-serif text-xl font-bold text-zinc-950">

                    {
                      selectedCharacter
                        .name
                        .charAt(0)
                    }

                  </div>


                  <p className="mt-5 text-xs font-medium uppercase tracking-[0.18em] text-amber-300">

                    Selected character

                  </p>


                  <h3 className="mt-2 font-serif text-2xl">

                    {
                      selectedCharacter.name
                    }

                  </h3>


                  <p className="mt-4 text-sm leading-6 text-zinc-400">

                    {
                      selectedCharacter.description
                      ||
                      "No description available."
                    }

                  </p>


                  <div className="my-6 border-t border-white/10" />


                  <p className="mb-4 text-xs font-medium uppercase tracking-[0.15em] text-zinc-500">

                    Relationships

                  </p>


                  <div className="space-y-3">

                    {
                      selectedRelationships
                        .slice(
                          0,
                          10
                        )
                        .map(
                          (
                            relationship,
                            index
                          ) => {

                            const other =
                              relationship.subject
                              === selected

                              ? relationship.object

                              : relationship.subject;


                            return (

                              <div
                                key={`${relationship.subject}-${relationship.object}-${index}`}

                                className="rounded-2xl border border-white/10 bg-white/[0.025] p-4"
                              >

                                <p className="text-xs text-amber-300">

                                  {
                                    prettyRelation(
                                      relationship.relation
                                    )
                                  }

                                </p>


                                <p className="mt-1 font-medium text-white">

                                  {other}

                                </p>


                                {
                                  relationship.explanation
                                  &&
                                  (

                                    <p className="mt-2 text-xs leading-5 text-zinc-500">

                                      {
                                        relationship.explanation
                                      }

                                    </p>

                                  )
                                }

                              </div>

                            );
                          }
                        )
                    }

                  </div>

                </>
              )

              : (

                <p className="text-sm text-zinc-500">

                  Select a character from the graph.

                </p>

              )
            }

          </aside>

        </div>

      </div>

    </main>
  );
}