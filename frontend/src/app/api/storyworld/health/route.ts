const API_URL =
  process.env.STORYWORLD_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API_URL}/api/health`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return Response.json(
        { status: "offline" },
        { status: response.status }
      );
    }

    const data = await response.json();

    return Response.json(data);
  } catch (error) {
    console.error("Health proxy error:", error);

    return Response.json(
      {
        status: "offline",
        detail: "Could not connect to FastAPI.",
      },
      {
        status: 503,
      }
    );
  }
}
