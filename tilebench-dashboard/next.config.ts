import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    useTypeScriptCli: false,
  },
  outputFileTracingIncludes: {
    // data/ is read at request time; make sure it ships with the functions.
    "/api/**": ["./data/**"],
  },
};

export default nextConfig;
