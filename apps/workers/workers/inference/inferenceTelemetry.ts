import {
  inferenceRequestCounter,
  inferenceRequestDurationHistogram,
} from "metrics";

import { addLogFields, setSpanAttributes } from "@karakeep/shared-server";
import serverConfig from "@karakeep/shared/config";
import type { InferenceClient } from "@karakeep/shared/inference";

import { InferenceRequestError, runInferenceRequest } from "./inferenceErrors";

/** Observe worker text/image requests without recording request or response content. */
export function withInferenceTelemetry(
  client: InferenceClient,
): InferenceClient {
  async function observe<T>(
    kind: "text" | "image",
    signal: AbortSignal | undefined,
    request: () => Promise<T>,
  ): Promise<T> {
    const model =
      kind === "text"
        ? serverConfig.inference.textModel
        : serverConfig.inference.imageModel;
    const start = performance.now();
    let outcome = "success";
    let status: number | undefined;
    try {
      return await runInferenceRequest(request, signal);
    } catch (error) {
      outcome =
        error instanceof InferenceRequestError ? error.code : "provider-error";
      status =
        error instanceof InferenceRequestError ? error.status : undefined;
      throw error;
    } finally {
      const seconds = (performance.now() - start) / 1000;
      // Labels come only from configuration and fixed enums, never captured content.
      const labels = { kind, model, outcome };
      inferenceRequestCounter.inc(labels);
      inferenceRequestDurationHistogram.observe(labels, seconds);
      addLogFields<"inferenceWorker.run">({
        "inference.model": model,
        "inference.request.kind": kind,
        "inference.request.duration": seconds,
        "inference.request.outcome": outcome,
        ...(status === undefined
          ? {}
          : { "inference.request.http_status": status }),
      });
      setSpanAttributes({
        "inference.model": model,
        "inference.request.kind": kind,
        "inference.request.duration": seconds,
        "inference.request.outcome": outcome,
        ...(status === undefined
          ? {}
          : { "inference.request.http_status": status }),
      });
    }
  }

  return {
    inferFromText: (prompt, opts) =>
      observe("text", opts.abortSignal, () =>
        client.inferFromText(prompt, opts),
      ),
    inferFromImage: (prompt, contentType, image, opts) =>
      observe("image", opts.abortSignal, () =>
        client.inferFromImage(prompt, contentType, image, opts),
      ),
    // Embedding has its own worker and is outside this text/image metric's scope.
    generateEmbeddingFromText: (inputs) =>
      client.generateEmbeddingFromText(inputs),
  };
}
