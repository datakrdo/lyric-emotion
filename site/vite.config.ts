import { defineConfig } from "vite";

// Served from https://<user>.github.io/lyric-emotion/
export default defineConfig({ base: "/lyric-emotion/", build: { target: "es2022" } });
