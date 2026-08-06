"""Exact-density LLM sidecar for calibrated SMC.

Serves three primitives over HTTP (stdlib only, plus llama-cpp-python):

  POST /encode   {text}                      -> {ids}
  POST /generate {prompt, max_tokens, seed?} -> {text, ids, logq_content, logq_eos, eos}
  POST /score    {prompt, ids}               -> {logq_content, logq_eos}

Semantics: temperature-1 sampling from the FULL softmax (no top-k/top-p/
penalties), so every reported log-probability is exactly the sampling
distribution, including the EOS step — closing the v1 gap where Ollama
omits the termination factor. /score teacher-forces an arbitrary token
sequence under an arbitrary prompt, which is the primitive that makes
feedback-conditioned MH kernels computable:
  alpha = min(1, [pi(x') q(x | prompt(x'))] / [pi(x) q(x' | prompt(x))]).

Atoms are token-id sequences: callers must check that generated ids equal
the tokenizer's canonical encoding of the canonical program text (the
designated atom), and /score must be called on exactly those ids.

Usage:
  .venv/bin/python llm_q_server.py --model <gguf path> --port 8765 \
      [--template chatml|raw] [--n-ctx 4096]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from llama_cpp import Llama


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    m = float(np.max(x))
    z = x - m
    return z - math.log(float(np.sum(np.exp(z))))


class Engine:
    def __init__(self, model_path: str, n_ctx: int, template: str) -> None:
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            logits_all=True,
            verbose=False,
        )
        self.template = template
        self.eos = self.llm.token_eos()
        self.n_ctx = n_ctx

    def render(self, prompt: str) -> list[int]:
        if self.template == "chatml":
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            text = prompt
        return self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)

    def encode(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)

    def _last_log_softmax(self, temperature: float) -> np.ndarray:
        # Read the last position's row from the preallocated numpy scores
        # buffer. Never touch eval_logits: it materializes EVERY position's
        # 152k-vocab logits as Python lists (~minutes per long prompt).
        # Temperature is applied BEFORE the softmax, so the returned values
        # are the exact log-density of the tempered sampling distribution;
        # /score with the same temperature reproduces it identically.
        logits = np.asarray(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float64)
        return log_softmax(logits / temperature)

    def generate(self, prompt: str, max_tokens: int, seed: int | None, temperature: float = 1.0) -> dict:
        rng = random.Random(seed)
        prompt_ids = self.render(prompt)
        self.llm.reset()
        self.llm.eval(prompt_ids)
        ids: list[int] = []
        logq_content = 0.0
        logq_eos = None
        hit_eos = False
        for _ in range(max_tokens):
            lsm = self._last_log_softmax(temperature)
            probs = np.exp(lsm)
            probs = probs / float(np.sum(probs))
            cumulative = np.cumsum(probs)
            token = int(np.searchsorted(cumulative, rng.random(), side="right"))
            token = min(token, len(probs) - 1)
            token_logq = float(lsm[token])
            if token == self.eos:
                logq_eos = token_logq
                hit_eos = True
                break
            ids.append(token)
            logq_content += token_logq
            if self.llm.n_tokens >= self.n_ctx - 1:
                break
            self.llm.eval([token])
        text = self.llm.detokenize(ids, special=False).decode("utf-8", errors="replace")
        return {
            "text": text,
            "ids": ids,
            "logq_content": logq_content,
            "logq_eos": logq_eos,
            "eos": hit_eos,
        }

    def score(self, prompt: str, ids: list[int], temperature: float = 1.0) -> dict:
        """Teacher-forced density with the SAME eval pattern as generation
        (prompt as one batch, then one token per eval): Metal kernels differ
        by batch shape, so only this pattern reproduces the sampling density
        exactly (verified max-abs-diff 0.0; batched scoring drifts ~6e-4)."""
        prompt_ids = self.render(prompt)
        if len(prompt_ids) + len(ids) + 1 >= self.n_ctx:
            raise ValueError("sequence exceeds context window")
        self.llm.reset()
        self.llm.eval(prompt_ids)
        logq_content = 0.0
        for token in ids:
            lsm = self._last_log_softmax(temperature)
            logq_content += float(lsm[int(token)])
            self.llm.eval([int(token)])
        final = self._last_log_softmax(temperature)
        return {"logq_content": logq_content, "logq_eos": float(final[self.eos])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--template", choices=["chatml", "raw"], default="chatml")
    arguments = parser.parse_args()
    engine = Engine(arguments.model, arguments.n_ctx, arguments.template)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._reply(200, {"ok": True, "model": arguments.model, "eos": engine.eos})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/encode":
                    self._reply(200, {"ids": engine.encode(request["text"])})
                elif self.path == "/generate":
                    self._reply(
                        200,
                        engine.generate(
                            request["prompt"],
                            int(request.get("max_tokens", 700)),
                            request.get("seed"),
                            float(request.get("temperature", 1.0)),
                        ),
                    )
                elif self.path == "/score":
                    self._reply(
                        200,
                        engine.score(
                            request["prompt"],
                            [int(i) for i in request["ids"]],
                            float(request.get("temperature", 1.0)),
                        ),
                    )
                else:
                    self._reply(404, {"error": "not found"})
            except Exception as error:  # noqa: BLE001 — surface everything to the client
                self._reply(500, {"error": str(error)})

    server = HTTPServer(("127.0.0.1", arguments.port), Handler)
    print(f"[sidecar] serving {arguments.model} on 127.0.0.1:{arguments.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
