const API_URL =
  process.env.STORYWORLD_API_URL ??
  "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(
      `${API_URL}/api/relationships`,
      {
        cache: "no-store",
      }
    );

    const data = await response.json();

    return Response.json(
      data,
      {
        status: response.status,
      }
    );
  } catch (error) {
    console.error(
      "Relationship proxy error:",
      error
    );

    return Response.json(
      {
        detail:
          "Could not connect to StoryWorld backend.",
      },
      {
        status: 503,
      }
    );
  }
}