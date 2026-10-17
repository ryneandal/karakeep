import { getEncoding } from "js-tiktoken";
import { describe, expect, it } from "vitest";

import { buildSummaryPrompt, buildTextPrompt } from "./prompts.server";

const encoding = getEncoding("o200k_base");
const builders = [
  {
    name: "tagging",
    build: (content: string, limit: number, custom: string[] = []) =>
      buildTextPrompt("english", custom, content, limit, "as-generated"),
  },
  {
    name: "summary",
    build: (content: string, limit: number, custom: string[] = []) =>
      buildSummaryPrompt("english", custom, content, limit),
  },
];

describe.each(builders)("$name prompt budget", ({ build }) => {
  it.each([
    "PostgreSQL recovery log. ".repeat(8000),
    "日本語・résumé・🦉 ".repeat(3000),
  ])("checks the complete prompt after truncating content", async (content) => {
    const prompt = await build(content, 2048);
    expect(encoding.encode(prompt).length).toBeLessThanOrEqual(2048);
    expect(prompt.length).toBeGreaterThan(0);
  });

  it("preserves short source content", async () => {
    const content = "PostgreSQL uses a write-ahead log for recovery.";
    expect(await build(content, 2048)).toContain(content);
  });

  it("rejects oversized instructions without disclosing their text", async () => {
    await expect(
      build("source", 2048, ["synthetic-private-canary ".repeat(3000)]),
    ).rejects.toThrow(
      /^Inference prompt instructions exceed INFERENCE_CONTEXT_LENGTH$/,
    );
  });

  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects an invalid token budget: %s",
    async (limit) => {
      await expect(build("source", limit)).rejects.toThrow(
        /^INFERENCE_CONTEXT_LENGTH must be a positive integer$/,
      );
    },
  );
});
