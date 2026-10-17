import { z } from "zod";

import { normalizeTagName } from "@karakeep/shared/utils/tag";

export const inferredTagsSchema = z.object({
  tags: z.array(z.string()),
});

export class InvalidTaggingResponseError extends Error {
  constructor() {
    // Do not retain the response, validation issues, or original error as a cause.
    super("Model returned an invalid structured tagging response");
    this.name = "InvalidTaggingResponseError";
  }
}

function parseJsonResponse(response: string): unknown {
  const trimmed = response.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const fenced = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/i.exec(trimmed);
    if (fenced) {
      try {
        return JSON.parse(fenced[1]);
      } catch {
        // Preserve the existing fallback for text surrounding a JSON object.
      }
    }
    const object = /\{[\s\S]*\}/.exec(trimmed);
    return JSON.parse(object ? object[0] : trimmed);
  }
}

export function parseTaggingResponse(response: string): string[] {
  try {
    return inferredTagsSchema
      .parse(parseJsonResponse(response))
      .tags.map((tag) => normalizeTagName(tag).trim());
  } catch {
    throw new InvalidTaggingResponseError();
  }
}

export function tagNormalizer() {
  // Keep this aligned with bookmarkTags.normalizedName in the database schema.
  return {
    normalizeTag: (tag: string) => tag.toLowerCase().replace(/[ \-_]/g, ""),
  };
}

export function filterInferredTags(
  inferredTags: readonly string[],
  excludedTags: readonly string[],
): { tags: string[]; rejectedCount: number } {
  const { normalizeTag } = tagNormalizer();
  const excluded = new Set(
    excludedTags.map((tag) => normalizeTag(normalizeTagName(tag).trim())),
  );
  const tags = inferredTags
    .map((tag) => normalizeTagName(tag).trim())
    .filter((tag) => tag.length > 0 && !excluded.has(normalizeTag(tag)));
  return { tags, rejectedCount: inferredTags.length - tags.length };
}
