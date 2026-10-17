import { describe, expect, it, vi } from "vitest";

import {
  filterInferredTags,
  InvalidTaggingResponseError,
  parseTaggingResponse,
  tagNormalizer,
} from "./taggingOutput";
import { runInferenceRequest, InferenceRequestError } from "./inferenceErrors";

const excluded = [
  "automation-ready",
  "automation-processing",
  "automation-consumed",
  "automation-error",
];

describe("provider error privacy", () => {
  it("does not start a request after cancellation", async () => {
    const controller = new AbortController();
    controller.abort("synthetic-private-abort-canary");
    const request = vi.fn(async () => "unused");
    await expect(
      runInferenceRequest(request, controller.signal),
    ).rejects.toMatchObject({ code: "aborted" });
    expect(request).not.toHaveBeenCalled();
  });

  it("preserves a successful provider response", async () => {
    const response = { response: '{"tags":["PostgreSQL"]}', tokensUsed: 12 };
    await expect(
      runInferenceRequest(async () => response, new AbortController().signal),
    ).resolves.toBe(response);
  });

  it.each([
    {
      original: new Error("synthetic-private-provider-canary"),
      status: undefined,
    },
    {
      original: Object.assign(new Error("synthetic-private-provider-canary"), {
        status: 503,
        headers: { authorization: "synthetic-private-provider-canary" },
      }),
      status: 503,
    },
    {
      original: {
        status: "synthetic-private-provider-canary",
        message: "synthetic-private-provider-canary",
      },
      status: undefined,
    },
    { original: "synthetic-private-provider-canary", status: undefined },
  ])(
    "retains only structural failure information",
    async ({ original, status }) => {
      let caught: unknown;
      try {
        await runInferenceRequest(async () => {
          throw original;
        }, new AbortController().signal);
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(InferenceRequestError);
      expect(String(caught)).not.toContain("synthetic-private-provider-canary");
      expect(JSON.stringify(caught)).not.toContain(
        "synthetic-private-provider-canary",
      );
      expect(caught).not.toHaveProperty("cause");
      expect(caught).not.toHaveProperty("headers");
      expect(caught).toHaveProperty("status", status);
      expect(caught).toHaveProperty(
        "code",
        status === undefined ? "provider-error" : "provider-http",
      );
    },
  );

  it("does not expose a cancellation reason", async () => {
    const controller = new AbortController();
    controller.abort("synthetic-private-abort-canary");
    let caught: unknown;
    try {
      await runInferenceRequest(async () => {
        throw controller.signal.reason;
      }, controller.signal);
    } catch (error) {
      caught = error;
    }
    expect(caught).toMatchObject({
      name: "InferenceRequestError",
      message: "Inference job failed (aborted)",
      code: "aborted",
    });
    expect(JSON.stringify(caught)).not.toContain(
      "synthetic-private-abort-canary",
    );
    expect(caught).not.toHaveProperty("cause");
  });
});

describe("inferred tag attachment policy", () => {
  it("rejects configured control tags supplied by model output", () => {
    const tags = parseTaggingResponse(
      '{"tags":["automation-ready","automation-consumed"]}',
    );
    expect(filterInferredTags(tags, excluded)).toEqual({
      tags: [],
      rejectedCount: 2,
    });
  });

  it.each([
    "automation-ready",
    "AUTOMATION_READY",
    "automation ready",
    "automationready",
    " ##Automation-Ready ",
  ])("rejects an excluded tag's matching alias: %s", (tag) => {
    expect(filterInferredTags([tag, "PostgreSQL"], excluded)).toEqual({
      tags: ["PostgreSQL"],
      rejectedCount: 1,
    });
  });

  it("preserves ordinary topics and does not reject substring matches", () => {
    expect(
      filterInferredTags(
        [" #PostgreSQL ", "architecture-automation-ready", "日本語"],
        excluded,
      ),
    ).toEqual({
      tags: ["PostgreSQL", "architecture-automation-ready", "日本語"],
      rejectedCount: 0,
    });
  });

  it("uses normalized configuration names and preserves comma-containing tags", () => {
    expect(
      filterInferredTags(
        ["sensitive-topic", "Budget, Groceries", "unrelated"],
        [" #Sensitive_Topic ", "budget, groceries"],
      ),
    ).toEqual({ tags: ["unrelated"], rejectedCount: 2 });
  });

  it("rejects empty normalized names even when exclusions are disabled", () => {
    expect(filterInferredTags(["", "  ", "###", "ordinary"], [])).toEqual({
      tags: ["ordinary"],
      rejectedCount: 3,
    });
  });

  it("keeps matching compatible with the existing database key", () => {
    const { normalizeTag } = tagNormalizer();
    expect(normalizeTag(" Automation Ready ")).toBe("automationready");
  });
});

describe("tagging response privacy and compatibility", () => {
  it.each([
    '{"tags":[" #PostgreSQL "]}',
    '```json\n{"tags":[" #PostgreSQL "]}\n```',
    'A model preamble: {"tags":[" #PostgreSQL "]} trailing text',
  ])("retains supported JSON wrappers", (response) => {
    expect(parseTaggingResponse(response)).toEqual(["PostgreSQL"]);
  });

  it("accepts an intentional empty tag result", () => {
    expect(parseTaggingResponse('{"tags":[]}')).toEqual([]);
  });

  it.each([
    "synthetic-private-canary is not JSON",
    '{"tags":[{"synthetic-private-canary":"value"}]}',
    '{"synthetic-private-canary":[]}',
    '{"tags":"synthetic-private-canary"}',
  ])(
    "does not expose model text or validation details in errors",
    (response) => {
      let caught: unknown;
      try {
        parseTaggingResponse(response);
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(InvalidTaggingResponseError);
      expect(String(caught)).not.toContain("synthetic-private-canary");
      expect(JSON.stringify(caught)).not.toContain("synthetic-private-canary");
      expect(caught).not.toHaveProperty("cause");
      expect(caught).not.toHaveProperty("issues");
    },
  );
});
