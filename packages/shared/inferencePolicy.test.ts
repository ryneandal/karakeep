import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("inference excluded-tag configuration", () => {
  it("defaults to an empty policy", async () => {
    vi.stubEnv("INFERENCE_TAGGING_EXCLUDED_TAGS", undefined);
    const { default: config } = await import("./config");
    expect(config.inference.excludedTags).toEqual([]);
  });

  it("validates and normalizes a JSON array using tag API semantics", async () => {
    vi.stubEnv(
      "INFERENCE_TAGGING_EXCLUDED_TAGS",
      '[" #Reserved_Tag ","budget, groceries"]',
    );
    const { default: config } = await import("./config");
    expect(config.inference.excludedTags).toEqual([
      "Reserved_Tag",
      "budget, groceries",
    ]);
  });

  it.each(["not JSON", "{}", "[42]", '[""]', '["###"]'])(
    "rejects invalid policy: %s",
    async (value) => {
      vi.stubEnv("INFERENCE_TAGGING_EXCLUDED_TAGS", value);
      await expect(import("./config")).rejects.toThrow();
    },
  );
});
