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
  generate(prompt: string, maxTokens?: number): Promise<SidecarDraw>;
  /** Exact log q of "emit exactly these ids then stop" under the prompt. */
  score(prompt: string, ids: readonly number[]): Promise<number>;
  encode(text: string): Promise<readonly number[]>;
}

export function createSidecarClient(baseUrl: string, requester: typeof fetch = fetch): SidecarClient {
  const root = baseUrl.replace(/\/$/, "");
  async function post<T>(path: string, body: unknown): Promise<T> {
    const response = await requester(`${root}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`sidecar ${path} HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
    }
    const payload = (await response.json()) as T & { error?: string };
    if (payload.error !== undefined) throw new Error(`sidecar ${path}: ${payload.error}`);
    return payload;
  }
  return {
    async generate(prompt, maxTokens = 700) {
      const result = await post<{
        text: string;
        ids: number[];
        logq_content: number;
        logq_eos: number | null;
        eos: boolean;
      }>("/generate", { prompt, max_tokens: maxTokens });
      return {
        text: result.text,
        ids: result.ids,
        // A draw that hit the length cap instead of EOS has no termination
        // factor; it can never be a designated atom, so -Infinity is correct.
        logQ: result.eos && result.logq_eos !== null ? result.logq_content + result.logq_eos : -Infinity,
        stoppedAtEos: result.eos,
      };
    },
    async score(prompt, ids) {
      const result = await post<{ logq_content: number; logq_eos: number }>("/score", {
        prompt,
        ids: [...ids],
      });
      return result.logq_content + result.logq_eos;
    },
    async encode(text) {
      const result = await post<{ ids: number[] }>("/encode", { text });
      return result.ids;
    },
  };
}
