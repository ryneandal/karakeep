import { eq } from "drizzle-orm";
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { InferenceClient } from "@karakeep/shared/inference";
import type { DequeuedJob } from "@karakeep/shared/queueing";
import type { ZOpenAIRequest } from "@karakeep/shared-server";
import { db } from "@karakeep/db";
import {
  bookmarkAssets,
  bookmarks,
  bookmarkTags,
  bookmarkTexts,
  tagsOnBookmarks,
  users,
} from "@karakeep/db/schema";
import { BookmarkTypes } from "@karakeep/shared/types/bookmarks";

import { runTagging } from "./tagging";
import {
  InvalidTaggingResponseError,
  TaggingInferenceError,
} from "./taggingOutput";

const effects = vi.hoisted(() => ({
  addLogFields: vi.fn(),
  setSpanAttributes: vi.fn(),
  triggerSearchReindex: vi.fn(),
  triggerWebhook: vi.fn(),
  triggerOnEvent: vi.fn(),
  debug: vi.fn(),
  info: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@karakeep/db", async () => {
  // Do not import drizzle.ts: its module initializer opens the configured disk DB.
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
        contextLength: 2048,
        excludedTags: ["automation-ready", "automation-consumed"],
      },
    },
  };
});

vi.mock("network", () => ({ getBookmarkDomain: () => undefined }));
vi.mock("trpc", () => ({
  buildImpersonatingTRPCClient: () => {
    throw new Error("Unexpected API call");
  },
}));
vi.mock("@karakeep/shared/vectorStore", () => ({
  getVectorStoreClient: () => null,
}));
vi.mock("@karakeep/shared/logger", () => ({ default: effects }));
vi.mock("@karakeep/shared-server", () => ({
  ...effects,
  ASSET_TYPES: { IMAGE_GIF: "image/gif" },
  readAsset: async () => ({
    asset: Buffer.from("synthetic-image-fixture"),
    metadata: { contentType: "image/png" },
  }),
}));
vi.mock("@karakeep/trpc/lib/ruleEngine", () => ({ RuleEngine: effects }));
vi.mock("@karakeep/trpc/models/bookmarks", () => ({
  Bookmark: {
    getBookmarkPlainTextContent: () => {
      throw new Error("Unexpected link read");
    },
  },
}));
vi.mock("@karakeep/trpc/models/webhooks.service", () => ({
  WebhooksService: class {
    triggerWebhook = effects.triggerWebhook;
  },
}));

const bookmarkId = "synthetic-bookmark";
const userId = "synthetic-user";
const originalText =
  "PostgreSQL uses a write-ahead log. synthetic-private-content-canary";
const client = {
  inferFromText: vi.fn<InferenceClient["inferFromText"]>(),
  inferFromImage: vi.fn<InferenceClient["inferFromImage"]>(),
  generateEmbeddingFromText:
    vi.fn<InferenceClient["generateEmbeddingFromText"]>(),
} satisfies InferenceClient;

function job(
  signal = new AbortController().signal,
): DequeuedJob<ZOpenAIRequest> {
  return {
    id: "synthetic-job",
    data: { bookmarkId, type: "tag" },
    priority: 1,
    runNumber: 1,
    abortSignal: signal,
  };
}

function attachedTags() {
  return db
    .select({ name: bookmarkTags.name, attachedBy: tagsOnBookmarks.attachedBy })
    .from(tagsOnBookmarks)
    .innerJoin(bookmarkTags, eq(tagsOnBookmarks.tagId, bookmarkTags.id))
    .where(eq(tagsOnBookmarks.bookmarkId, bookmarkId))
    .orderBy(bookmarkTags.name)
    .all();
}

function expectCapturePreserved() {
  expect(db.select().from(bookmarkTexts).get()?.text).toBe(originalText);
  expect(db.select().from(bookmarks).all()).toHaveLength(1);
  expect(attachedTags()).toEqual([
    { name: "Existing topic", attachedBy: "ai" },
    { name: "automation-ready", attachedBy: "human" },
  ]);
}

function expectNoPrivateLogs() {
  for (const mock of [
    effects.debug,
    effects.info,
    effects.error,
    effects.addLogFields,
    effects.setSpanAttributes,
  ]) {
    expect(JSON.stringify(mock.mock.calls)).not.toContain("canary");
  }
}

beforeEach(() => {
  vi.resetAllMocks();
  db.delete(users).run();
  db.insert(users)
    .values({ id: userId, name: "Synthetic", email: "fixture@example.invalid" })
    .run();
  db.insert(bookmarks)
    .values({ id: bookmarkId, type: BookmarkTypes.TEXT, userId })
    .run();
  db.insert(bookmarkTexts).values({ id: bookmarkId, text: originalText }).run();
  db.insert(bookmarkTags)
    .values([
      { id: "ready", name: "automation-ready", userId },
      { id: "consumed", name: "automation-consumed", userId },
      { id: "existing", name: "Existing topic", userId },
    ])
    .run();
  db.insert(tagsOnBookmarks)
    .values([
      { bookmarkId, tagId: "ready", attachedBy: "human" },
      { bookmarkId, tagId: "existing", attachedBy: "ai" },
    ])
    .run();
});

afterAll(() => db.$client.close());

describe("tagging worker attachment boundary", () => {
  it("attaches ordinary topics without granting model output lifecycle authority", async () => {
    client.inferFromText.mockResolvedValue({
      response:
        '{"tags":["AUTOMATION_READY","automation-consumed","PostgreSQL"]}',
      totalTokens: 20,
    });
    await runTagging(bookmarkId, job(), client);
    expect(attachedTags()).toEqual([
      { name: "PostgreSQL", attachedBy: "ai" },
      { name: "automation-ready", attachedBy: "human" },
    ]);
    expect(db.select().from(bookmarkTags).all()).toHaveLength(4);
    expect(effects.addLogFields).toHaveBeenCalledWith({
      "inference.tagging.num_rejected_tags": 2,
    });
    expect(effects.triggerOnEvent).toHaveBeenCalledOnce();
    expect(effects.triggerWebhook).toHaveBeenCalledOnce();
    expect(effects.triggerSearchReindex).toHaveBeenCalledOnce();
    expectNoPrivateLogs();
  });

  it("leaves every existing association intact when all proposed tags are excluded", async () => {
    client.inferFromText.mockResolvedValue({
      response: '{"tags":["automation-ready","automation consumed"]}',
      totalTokens: 10,
    });
    await runTagging(bookmarkId, job(), client);
    expectCapturePreserved();
    expect(db.select().from(bookmarkTags).all()).toHaveLength(3);
    expect(effects.triggerOnEvent).not.toHaveBeenCalled();
    expectNoPrivateLogs();
  });

  it("preserves capture and associations after an invalid private model response", async () => {
    client.inferFromText.mockResolvedValue({
      response: "synthetic-private-response-canary",
      totalTokens: 7,
    });
    await expect(runTagging(bookmarkId, job(), client)).rejects.toBeInstanceOf(
      InvalidTaggingResponseError,
    );
    expectCapturePreserved();
    expect(effects.triggerWebhook).not.toHaveBeenCalled();
    expect(effects.triggerSearchReindex).not.toHaveBeenCalled();
    expectNoPrivateLogs();
  });

  it("preserves capture and reports only HTTP status when the provider fails", async () => {
    client.inferFromText.mockRejectedValue(
      Object.assign(new Error("synthetic-private-provider-canary"), {
        status: 503,
      }),
    );
    await expect(runTagging(bookmarkId, job(), client)).rejects.toMatchObject({
      name: "TaggingInferenceError",
      status: 503,
      code: "provider-http",
      message: "Tagging inference request failed with HTTP 503",
    });
    expectCapturePreserved();
    expect(effects.triggerWebhook).not.toHaveBeenCalled();
    expectNoPrivateLogs();
  });

  it("forwards cancellation and preserves capture without logging the abort reason", async () => {
    const controller = new AbortController();
    controller.abort("synthetic-private-abort-canary");
    client.inferFromText.mockRejectedValue(controller.signal.reason);
    await expect(
      runTagging(bookmarkId, job(controller.signal), client),
    ).rejects.toMatchObject({ code: "aborted" });
    expect(client.inferFromText.mock.calls[0]?.[1].abortSignal).toBe(
      controller.signal,
    );
    expectCapturePreserved();
    expectNoPrivateLogs();
  });

  it("bounds the default prompt while retaining the full captured text", async () => {
    const text = `${"PostgreSQL recovery log. ".repeat(8000)}synthetic-private-tail-canary`;
    db.update(bookmarkTexts)
      .set({ text })
      .where(eq(bookmarkTexts.id, bookmarkId))
      .run();
    client.inferFromText.mockResolvedValue({
      response: '{"tags":[]}',
      totalTokens: 5,
    });
    await runTagging(bookmarkId, job(), client);
    const prompt = client.inferFromText.mock.calls[0]?.[0];
    expect(prompt).toBeDefined();
    expect(prompt).not.toContain("synthetic-private-tail-canary");
    const { getEncoding } = await import("js-tiktoken");
    expect(
      getEncoding("o200k_base").encode(prompt!).length,
    ).toBeLessThanOrEqual(2048);
    expect(db.select().from(bookmarkTexts).get()?.text).toBe(text);
    expectNoPrivateLogs();
  });

  it.each(["image", "pdf"] as const)(
    "redacts provider errors on the %s path",
    async (assetType) => {
      db.delete(bookmarkTexts).run();
      db.update(bookmarks)
        .set({ type: BookmarkTypes.ASSET })
        .where(eq(bookmarks.id, bookmarkId))
        .run();
      db.insert(bookmarkAssets)
        .values({
          id: bookmarkId,
          assetId: "synthetic-asset",
          assetType,
          content: "synthetic-private-asset-canary",
        })
        .run();
      client.inferFromText.mockRejectedValue(
        new Error("synthetic-private-provider-canary"),
      );
      client.inferFromImage.mockRejectedValue(
        new Error("synthetic-private-provider-canary"),
      );
      await expect(
        runTagging(bookmarkId, job(), client),
      ).rejects.toBeInstanceOf(TaggingInferenceError);
      expect(db.select().from(bookmarkAssets).get()?.content).toBe(
        "synthetic-private-asset-canary",
      );
      expect(effects.triggerWebhook).not.toHaveBeenCalled();
      expectNoPrivateLogs();
    },
  );
});
