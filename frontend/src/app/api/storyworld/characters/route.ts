const API_URL =
  process.env.STORYWORLD_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_URL}/api/characters`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return Response.json(
        { detail: "Could not load characters." },
        { status: response.status }
      );
    }

    return Response.json(await response.json());
  } catch {
    return Response.json(
      { detail: "StoryWorld backend unavailable." },
      { status: 503 }
    );
  }
}