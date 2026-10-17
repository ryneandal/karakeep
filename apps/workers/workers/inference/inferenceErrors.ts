export type InferenceErrorCode =
  | "aborted"
  | "provider-http"
  | "provider-error"
  | "invalid-response"
  | "invalid-job"
  | "job-error";

export class InferenceJobError extends Error {
  constructor(
    readonly code: InferenceErrorCode,
    readonly status?: number,
  ) {
    super(
      `Inference job failed (${code})${status === undefined ? "" : `: HTTP ${status}`}`,
    );
    this.name = "InferenceJobError";
  }
}

export class InferenceRequestError extends InferenceJobError {
  constructor(status: number | undefined, aborted: boolean) {
    super(
      aborted
        ? "aborted"
        : status === undefined
          ? "provider-error"
          : "provider-http",
      status,
    );
    this.name = "InferenceRequestError";
  }
}

export async function runInferenceRequest<T>(
  request: () => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  try {
    if (signal?.aborted) {
      throw new InferenceRequestError(undefined, true);
    }
    const response = await request();
    // A client may resolve with a partial response after cancellation.
    if (signal?.aborted) {
      throw new InferenceRequestError(undefined, true);
    }
    return response;
  } catch (error) {
    const candidate =
      typeof error === "object" && error !== null && "status" in error
        ? error.status
        : undefined;
    const status =
      typeof candidate === "number" &&
      Number.isInteger(candidate) &&
      candidate >= 100 &&
      candidate <= 599
        ? candidate
        : undefined;
    // Never retain provider messages, bodies, headers, causes, or abort reasons.
    throw new InferenceRequestError(status, signal?.aborted ?? false);
  }
}
