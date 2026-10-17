import { beforeEach, describe, expect, it, vi } from "vitest";

import { InferenceClientFactory } from "./inference";

const provider = vi.hoisted(() => ({
  generate: vi.fn(),
  abort: vi.fn(),
  warn: vi.fn(),
}));

vi.mock("ollama", () => ({
  Ollama: class {
    generate = provider.generate;
    abort = provider.abort;
  },
}));
vi.mock("./logger", () => ({ default: { warn: provider.warn } }));
vi.mock("./config", () => ({
  default: {
    inference: {
      ollamaBaseUrl: "http://127.0.0.1:11434",
      textModel: "synthetic-text-route",
      imageModel: "synthetic-image-route",
      contextLength: 2048,
      maxOutputTokens: 512,
      outputSchema: "plain",
    },
  },
}));

beforeEach(() => vi.resetAllMocks());

describe("Ollama partial-stream privacy", () => {
  it.each(["text", "image"])(
    "does not log provider errors or partial %s output",
    async (kind) => {
      async function* partialResponse() {
        yield {
          response: "synthetic-private-output-canary",
          eval_count: 2,
          prompt_eval_count: 3,
        };
        throw new Error("synthetic-private-provider-canary");
      }
      provider.generate.mockResolvedValue(partialResponse());
      const client = InferenceClientFactory.build();
      expect(client).not.toBeNull();
      const result =
        kind === "text"
          ? await client!.inferFromText("synthetic-private-prompt-canary", {})
          : await client!.inferFromImage(
              "synthetic-private-prompt-canary",
              "image/png",
              "synthetic-image",
              {},
            );
      expect(result).toEqual({
        response: "synthetic-private-output-canary",
        totalTokens: Number.NaN,
      });
      expect(provider.warn).toHaveBeenCalledOnce();
      expect(JSON.stringify(provider.warn.mock.calls)).not.toContain("canary");
    },
  );

  it("preserves cancellation instead of accepting a partial response", async () => {
    const error = new Error("synthetic-private-abort-canary");
    error.name = "AbortError";
    async function* canceledResponse() {
      yield { response: "partial", eval_count: 1, prompt_eval_count: 2 };
      throw error;
    }
    provider.generate.mockResolvedValue(canceledResponse());
    const client = InferenceClientFactory.build();
    await expect(client!.inferFromText("synthetic", {})).rejects.toBe(error);
    expect(provider.warn).not.toHaveBeenCalled();
  });
});
