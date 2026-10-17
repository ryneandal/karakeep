import type { Tiktoken } from "js-tiktoken";

import type { ZTagStyle } from "./types/users";
import { constructSummaryPrompt, constructTextTaggingPrompt } from "./prompts";

let encoding: Tiktoken | null = null;

/**
 * Lazy load the encoding to avoid loading the tiktoken data into memory
 * until it's actually needed
 */
async function getEncodingInstance(): Promise<Tiktoken> {
  if (!encoding) {
    // Dynamic import to lazy load the tiktoken module
    const { getEncoding } = await import("js-tiktoken");
    encoding = getEncoding("o200k_base");
  }
  return encoding;
}

async function buildBoundedPrompt(
  content: string,
  contextLength: number,
  render: (content: string) => string,
): Promise<string> {
  if (!Number.isInteger(contextLength) || contextLength <= 0) {
    throw new Error("INFERENCE_CONTEXT_LENGTH must be a positive integer");
  }
  const enc = await getEncodingInstance();
  const template = render("");
  const templateSize = enc.encode(template).length;
  if (templateSize > contextLength) {
    // Do not truncate trusted instructions or include custom prompt text in errors.
    throw new Error(
      "Inference prompt instructions exceed INFERENCE_CONTEXT_LENGTH",
    );
  }
  const tokens = enc.encode(content);
  let available = Math.min(tokens.length, contextLength - templateSize);
  while (available > 0) {
    const prompt = render(
      available === tokens.length
        ? content
        : enc.decode(tokens.slice(0, available)),
    );
    const overflow = enc.encode(prompt).length - contextLength;
    if (overflow <= 0) {
      return prompt;
    }
    // Token boundaries can change when content is inserted into the template.
    // Recheck the assembled prompt and strictly reduce the content each time.
    available = Math.max(0, available - overflow);
  }
  return template;
}

/**
 * Remove duplicate whitespaces to avoid tokenization issues
 */
function preprocessContent(content: string) {
  return content.replace(/(\s){10,}/g, "$1");
}

export async function buildTextPrompt(
  lang: string,
  customPrompts: string[],
  content: string,
  contextLength: number,
  tagStyle: ZTagStyle,
  curatedTags?: string[],
  potentialRelevantTags?: string[],
): Promise<string> {
  return buildBoundedPrompt(
    preprocessContent(content),
    contextLength,
    (boundedContent) =>
      constructTextTaggingPrompt(
        lang,
        customPrompts,
        boundedContent,
        tagStyle,
        curatedTags,
        potentialRelevantTags,
      ),
  );
}

export async function buildSummaryPrompt(
  lang: string,
  customPrompts: string[],
  content: string,
  contextLength: number,
): Promise<string> {
  return buildBoundedPrompt(
    preprocessContent(content),
    contextLength,
    (boundedContent) =>
      constructSummaryPrompt(lang, customPrompts, boundedContent),
  );
}
