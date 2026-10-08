import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TileBench++ — GPU Kernel DSL Benchmark",
  description:
    "TileBench++ benchmarks Triton, cuTile and TileLang against torch across 45 GPU kernels on NVIDIA B200, NVIDIA GH200 and AMD MI300X, with profiling reports, kernel source and a performance query agent.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
