const API_URL =
  process.env.STORYWORLD_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_URL}/api/story`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return Response.json(
        { detail: "StoryWorld backend unavailable." },
        { status: response.status }
      );
    }

    const data = await response.json();

    return Response.json(data);
  } catch {
    return Response.json(
      { detail: "Could not connect to StoryWorld backend." },
      { status: 503 }
    );
  }
}