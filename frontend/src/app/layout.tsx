import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StoryWorld",
  description:
    "Explore stories through grounded AI reasoning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}