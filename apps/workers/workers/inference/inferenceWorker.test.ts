import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import type { ZOpenAIRequest } from "@karakeep/shared-server";
import type { InferenceClient } from "@karakeep/shared/inference";
import type { DequeuedJob, RunnerFuncs } from "@karakeep/shared/queueing";
import { db } from "@karakeep/db";
import {
  bookmarkLinks,
  bookmarks,
  bookmarkTags,
  tagsOnBookmarks,
  users,
} from "@karakeep/db/schema";
import { BookmarkTypes } from "@karakeep/shared/types/bookmarks";

import { initEventLogger } from "../../../../packages/shared-server/src/eventLogger";
import { registry } from "../../metrics";
import { withInferenceTelemetry } from "./inferenceTelemetry";
import { OpenAiWorker } from "./inferenceWorker";

const effects = vi.hoisted(() => ({
  build: vi.fn(),
  createRunner:
    vi.fn<(queue: unknown, callbacks: RunnerFuncs<ZOpenAIRequest>) => void>(),
  event: vi.fn(),
  spanStatus: vi.fn(),
  spanError: vi.fn(),
  spanAttributes: vi.fn(),
  spanEnd: vi.fn(),
  triggerSearchReindex: vi.fn(),
  triggerWebhook: vi.fn(),
  triggerOnEvent: vi.fn(),
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@karakeep/db", async () => {
  // Use real migrations and queries without opening the operator's configured DB.
  const { default: Database } = await import("better-sqlite3");
  const { drizzle } = await import("drizzle-orm/better-sqlite3");
  const { migrate } = await import("drizzle-orm/better-sqlite3/migrator");
  const { fileURLToPath } = await import("node:url");
  const schema = await import("@karakeep/db/schema");
  const sqlite = new Database(":memory:");
  sqlite.pragma("foreign_keys = ON");
  const db = drizzle(sqlite, { schema });
  migrate(db, {
    migrationsFolder: fileURLToPath(
      new URL("../../../../packages/db/drizzle", import.meta.url),
    ),
  });
  return { db };
});

vi.mock("@karakeep/shared/config", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@karakeep/shared/config")>();
  return {
    ...actual,
    default: {
      ...actual.default,
      inference: {
        ...actual.default.inference,
        enableAutoTagging: true,
        enableAutoSummarization: true,
        contextLength: 2048,
        textModel: "test-text-route",
        imageModel: "test-image-route",
      },
      prometheus: { enabled: false },
      eventLogs: { enabled: true, otlpExport: { enabled: false } },
    },
  };
});
vi.mock("@karakeep/shared/logger", () => ({ default: effects }));
vi.mock("@karakeep/shared/inference", () => ({
  InferenceClientFactory: { build: effects.build },
}));
vi.mock("@karakeep/shared/queueing", () => ({
  getQueueClient: async () => ({ createRunner: effects.createRunner }),
}));
vi.mock("winston", () => ({
  default: {
    createLogger: () => ({ log: effects.event }),
    format: { combine: vi.fn(), timestamp: vi.fn(), json: vi.fn() },
    transports: { Console: class {} },
  },
}));
vi.mock("@karakeep/shared-server", async () => {
  const events =
    await import("../../../../packages/shared-server/src/eventLogger");
  const { withSpan } =
    await import("../../../../packages/shared-server/src/tracing");
  const queues = await import("../../../../packages/shared-server/src/queues");
  const span = {
    setStatus: effects.spanStatus,
    recordException: effects.spanError,
    end: effects.spanEnd,
  };
  return {
    ...events,
    ...queues,
    withSpan,
    getTracer: () => ({
      startActiveSpan: (
        _name: string,
        _options: unknown,
        fn: (value: typeof span) => unknown,
      ) => fn(span),
    }),
    setSpanAttributes: effects.spanAttributes,
    triggerSearchReindex: effects.triggerSearchReindex,
    ASSET_TYPES: { IMAGE_GIF: "image/gif" },
    readAsset: vi.fn(),
  };
});
vi.mock("trpc", () => ({ buildImpersonatingTRPCClient: vi.fn() }));
vi.mock("@karakeep/shared/vectorStore", () => ({
  getVectorStoreClient: () => null,
}));
vi.mock("@karakeep/trpc/lib/ruleEngine", () => ({ RuleEngine: effects }));
vi.mock("@karakeep/trpc/models/bookmarks", () => ({
  Bookmark: {
    getBookmarkPlainTextContent: () =>
      "PostgreSQL stores data. private-content-canary",
  },
}));
vi.mock("@karakeep/trpc/models/webhooks.service", () => ({
  WebhooksService: class {
    triggerWebhook = effects.triggerWebhook;
  },
}));

const bookmarkId = "synthetic-bookmark";
const url = "https://private-url-canary.invalid/private-path-canary";
const htmlContent = "<p>private-content-canary</p>";
const oldSummary = "private-existing-summary-canary";
const client = {
  inferFromText: vi.fn<InferenceClient["inferFromText"]>(),
  inferFromImage: vi.fn<InferenceClient["inferFromImage"]>(),
  generateEmbeddingFromText:
    vi.fn<InferenceClient["generateEmbeddingFromText"]>(),
} satisfies InferenceClient;
let callbacks: RunnerFuncs<ZOpenAIRequest>;

function job(
  type: ZOpenAIRequest["type"] = "summarize",
  signal = new AbortController().signal,
): DequeuedJob<ZOpenAIRequest> {
  return {
    id: "synthetic-job",
    data: { bookmarkId, type },
    priority: 1,
    runNumber: 1,
    abortSignal: signal,
  };
}

function expectCapturePreserved() {
  expect(db.select().from(bookmarkLinks).get()).toMatchObject({
    url,
    htmlContent,
  });
  expect(db.select().from(bookmarks).all()).toHaveLength(1);
  expect(db.select().from(tagsOnBookmarks).all()).toMatchObject([
    { bookmarkId, tagId: "ready", attachedBy: "human" },
  ]);
}

async function expectPrivateDataAbsent(error?: unknown) {
  const observed = [error, ...effects.spanError.mock.calls.flat()];
  for (const item of observed) {
    if (item instanceof Error) {
      expect(
        `${item.message}\n${item.stack}\n${JSON.stringify(item)}`,
      ).not.toContain("canary");
      expect(item).not.toHaveProperty("cause");
    }
  }
  for (const mock of [
    effects.event,
    effects.spanStatus,
    effects.spanAttributes,
    effects.debug,
    effects.info,
    effects.warn,
    effects.error,
  ]) {
    expect(JSON.stringify(mock.mock.calls)).not.toContain("canary");
  }
  expect(await registry.metrics()).not.toContain("canary");
}

beforeAll(async () => {
  await initEventLogger("test");
});
beforeEach(async () => {
  vi.resetAllMocks();
  registry.resetMetrics();
  effects.build.mockReturnValue(client);
  db.delete(users).run();
  db.insert(users)
    .values({
      id: "synthetic-user",
      name: "Synthetic",
      email: "fixture@example.invalid",
    })
    .run();
  db.insert(bookmarks)
    .values({
      id: bookmarkId,
      type: BookmarkTypes.LINK,
      userId: "synthetic-user",
      summary: oldSummary,
    })
    .run();
  db.insert(bookmarkLinks)
    .values({
      id: bookmarkId,
      url,
      title: "private-title-canary",
      htmlContent,
      description: "private-description-canary",
    })
    .run();
  db.insert(bookmarkTags)
    .values({ id: "ready", name: "automation-ready", userId: "synthetic-user" })
    .run();
  db.insert(tagsOnBookmarks)
    .values({ bookmarkId, tagId: "ready", attachedBy: "human" })
    .run();
  await OpenAiWorker.build();
  callbacks = effects.createRunner.mock.calls[0][1];
});
afterAll(() => db.$client.close());

describe("inference worker telemetry and failure boundary", () => {
  it("preserves capture and summary through a 503 and exhausted retries without exposing provider data", async () => {
    client.inferFromText.mockRejectedValue(
      Object.assign(new Error("private-provider-canary"), {
        status: 503,
        cause: "private-cause-canary",
        headers: { authorization: "private-credential-canary" },
      }),
    );
    const error = await callbacks.run(job()).catch((value: unknown) => value);
    expect(error).toMatchObject({
      name: "InferenceJobError",
      code: "provider-http",
      status: 503,
    });
    expect(effects.event).toHaveBeenCalledWith(
      "error",
      expect.objectContaining({
        "event.name": "inferenceWorker.run",
        "inference.model": "test-text-route",
        "inference.request.kind": "text",
        "inference.request.outcome": "provider-http",
        "inference.request.http_status": 503,
        "inference.error.code": "provider-http",
        "inference.request.duration": expect.any(Number),
      }),
    );
    expect(effects.spanError).toHaveBeenCalledWith(error);
    expect(effects.spanEnd).toHaveBeenCalledOnce();
    const metrics = await registry.metrics();
    expect(metrics).toContain(
      'karakeep_inference_requests_total{kind="text",model="test-text-route",outcome="provider-http"} 1',
    );
    expect(metrics).toContain(
      'karakeep_inference_request_duration_seconds_count{kind="text",model="test-text-route",outcome="provider-http"} 1',
    );
    if (!(error instanceof Error)) throw new Error("Expected worker error");
    await callbacks.onError!({ ...job(), error, numRetriesLeft: 2 });
    expect(db.select().from(bookmarks).get()?.summarizationStatus).toBe(
      "pending",
    );
    await callbacks.onError!({ ...job(), error, numRetriesLeft: 0 });
    expect(db.select().from(bookmarks).get()).toMatchObject({
      summary: oldSummary,
      summarizationStatus: "failure",
    });
    expectCapturePreserved();
    expect(effects.triggerSearchReindex).not.toHaveBeenCalled();
    await expectPrivateDataAbsent(error);
  });

  it("records successful summary requests without logging source or generated content", async () => {
    client.inferFromText.mockResolvedValue({
      response: "private-generated-summary-canary",
      totalTokens: 12,
    });
    await callbacks.run(job());
    await callbacks.onComplete!(job(), undefined);
    expect(db.select().from(bookmarks).get()).toMatchObject({
      summary: "private-generated-summary-canary",
      summarizationStatus: "success",
    });
    expectCapturePreserved();
    expect(effects.triggerSearchReindex).toHaveBeenCalledOnce();
    expect(effects.event).toHaveBeenCalledWith(
      "info",
      expect.objectContaining({
        "inference.request.outcome": "success",
        "inference.total_tokens": 12,
      }),
    );
    expect(await registry.metrics()).toContain(
      'karakeep_inference_requests_total{kind="text",model="test-text-route",outcome="success"} 1',
    );
    await expectPrivateDataAbsent();
  });

  it("records an invalid tagging response as a job failure after a successful transport request", async () => {
    client.inferFromText.mockResolvedValue({
      response: "private-model-response-canary",
      totalTokens: 3,
    });
    await expect(callbacks.run(job("tag"))).rejects.toMatchObject({
      code: "invalid-response",
    });
    expect(effects.event).toHaveBeenCalledWith(
      "error",
      expect.objectContaining({
        "inference.request.outcome": "success",
        "inference.error.code": "invalid-response",
      }),
    );
    expectCapturePreserved();
    await expectPrivateDataAbsent();
  });

  it("records cancellation without exposing the abort reason", async () => {
    const controller = new AbortController();
    client.inferFromText.mockImplementation(async () => {
      controller.abort("private-abort-canary");
      throw controller.signal.reason;
    });
    await expect(
      callbacks.run(job("summarize", controller.signal)),
    ).rejects.toMatchObject({ code: "aborted" });
    expect(await registry.metrics()).toContain('outcome="aborted"} 1');
    expectCapturePreserved();
    await expectPrivateDataAbsent();
  });

  it.each(["summarize", "tag"] as const)(
    "discards a late %s response after cancellation",
    async (type) => {
      const controller = new AbortController();
      client.inferFromText.mockImplementation(async () => {
        controller.abort("private-abort-canary");
        return {
          response:
            type === "tag"
              ? '{"tags":["private-late-tag-canary"]}'
              : "private-late-summary-canary",
          totalTokens: 4,
        };
      });
      await expect(
        callbacks.run(job(type, controller.signal)),
      ).rejects.toMatchObject({ code: "aborted" });
      expect(db.select().from(bookmarks).get()?.summary).toBe(oldSummary);
      expectCapturePreserved();
      expect(effects.triggerSearchReindex).not.toHaveBeenCalled();
      expect(await registry.metrics()).toContain(
        'karakeep_inference_requests_total{kind="text",model="test-text-route",outcome="aborted"} 1',
      );
      await expectPrivateDataAbsent();
    },
  );

  it("sanitizes failures before a provider request reaches event logging and tracing", async () => {
    effects.build.mockImplementation(() => {
      throw new Error("private-config-canary");
    });
    await expect(callbacks.run(job())).rejects.toMatchObject({
      code: "job-error",
    });
    expect(client.inferFromText).not.toHaveBeenCalled();
    expect(effects.event).toHaveBeenCalledWith(
      "error",
      expect.objectContaining({ "inference.error.code": "job-error" }),
    );
    expect(effects.spanError).toHaveBeenCalledOnce();
    expectCapturePreserved();
    await expectPrivateDataAbsent();
  });

  it("does not expose malformed job fields through validation errors or status updates", async () => {
    const invalid = {
      ...job(),
      data: JSON.parse('{"type":"private-job-canary"}'),
    };
    await expect(callbacks.run(invalid)).rejects.toMatchObject({
      code: "invalid-job",
    });
    await callbacks.onError!({
      ...invalid,
      error: new Error("private-queue-canary"),
      numRetriesLeft: 0,
    });
    expect(effects.error).toHaveBeenCalledWith(
      "Failed to update inference job status",
    );
    expectCapturePreserved();
    await expectPrivateDataAbsent();
  });

  it("sanitizes errors supplied by the queue independently of the run callback", async () => {
    await callbacks.onError!({
      ...job(),
      error: new Error("private-queue-stack-canary"),
      numRetriesLeft: 0,
    });
    expect(db.select().from(bookmarks).get()?.summarizationStatus).toBe(
      "failure",
    );
    expect(effects.error).toHaveBeenCalledWith(
      "[inference][synthetic-job] Inference job failed (job-error)",
    );
    expectCapturePreserved();
    await expectPrivateDataAbsent();
  });

  it.each([true, false])(
    "uses the image route for image request metrics (failure=%s)",
    async (failed) => {
      const result = {
        response: "private-image-response-canary",
        totalTokens: 5,
      };
      if (failed)
        client.inferFromImage.mockRejectedValue(
          new Error("private-image-error-canary"),
        );
      else client.inferFromImage.mockResolvedValue(result);
      const call = withInferenceTelemetry(client).inferFromImage(
        "private-prompt-canary",
        "image/png",
        "private-image-canary",
        {},
      );
      if (failed)
        await expect(call).rejects.toMatchObject({ code: "provider-error" });
      else await expect(call).resolves.toBe(result);
      const outcome = failed ? "provider-error" : "success";
      expect(await registry.metrics()).toContain(
        `karakeep_inference_requests_total{kind="image",model="test-image-route",outcome="${outcome}"} 1`,
      );
      await expectPrivateDataAbsent();
    },
  );
});
