/**
 * Client for the exact-density LLM sidecar (experiments/sidecar/llm_q_server.py).
 *
 * The sidecar samples at temperature 1 from the full softmax and reports every
 * step's log-probability INCLUDING the EOS step, and can teacher-force score an
 * arbitrary token sequence under an arbitrary prompt with the same eval pattern
 * as generation — the primitives for feedback-conditioned MH kernels.
 */

export interface SidecarDraw {
  readonly text: string;
  readonly ids: readonly number[];
  /** Sum of content-token log-probs plus the EOS log-prob (exact q of "emit text then stop"). */
  readonly logQ: number;
  readonly stoppedAtEos: boolean;
}

export interface SidecarClient {
  generate(prompt: string, maxTokens?: number, temperature?: number): Promise<SidecarDraw>;
  /** Exact log q of "emit exactly these ids then stop" under the prompt. */
  score(prompt: string, ids: readonly number[], temperature?: number): Promise<number>;
  encode(text: string): Promise<readonly number[]>;
}

export interface SidecarClientOptions {
  /** Per-request timeout in ms (default 120000). A stalled request aborts and is retried. */
  readonly timeoutMs?: number;
  /** How many times to retry a failed request before giving up (default 3). */
  readonly retries?: number;
}

export function createSidecarClient(
  baseUrl: string,
  requester: typeof fetch = fetch,
  options: SidecarClientOptions = {},
): SidecarClient {
  const root = baseUrl.replace(/\/$/, "");
  const timeoutMs = options.timeoutMs ?? 120_000;
  const retries = options.retries ?? 3;
  async function post<T>(path: string, body: unknown): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await requester(`${root}${path}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`sidecar ${path} HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
        }
        const payload = (await response.json()) as T & { error?: string };
        if (payload.error !== undefined) throw new Error(`sidecar ${path}: ${payload.error}`);
        return payload;
      } catch (error) {
        // A 500 with a server-reported error is deterministic — don't retry it.
        if (error instanceof Error && error.message.includes(`sidecar ${path}:`)) throw error;
        lastError = error;
      } finally {
        clearTimeout(timer);
      }
    }
    throw new Error(
      `sidecar ${path} failed after ${retries + 1} attempts: ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }
  return {
    async generate(prompt, maxTokens = 700, temperature = 1) {
      const result = await post<{
        text: string;
        ids: number[];
        logq_content: number;
        logq_eos: number | null;
        eos: boolean;
      }>("/generate", { prompt, max_tokens: maxTokens, temperature });
      return {
        text: result.text,
        ids: result.ids,
        // A draw that hit the length cap instead of EOS has no termination
        // factor; it can never be a designated atom, so -Infinity is correct.
        logQ: result.eos && result.logq_eos !== null ? result.logq_content + result.logq_eos : -Infinity,
        stoppedAtEos: result.eos,
      };
    },
    async score(prompt, ids, temperature = 1) {
      const result = await post<{ logq_content: number; logq_eos: number }>("/score", {
        prompt,
        ids: [...ids],
        temperature,
      });
      return result.logq_content + result.logq_eos;
    },
    async encode(text) {
      const result = await post<{ ids: number[] }>("/encode", { text });
      return result.ids;
    },
  };
}
