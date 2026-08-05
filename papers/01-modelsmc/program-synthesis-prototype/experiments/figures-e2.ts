/**
 * Experiment-2 figure: exact-solve rate across the task-difficulty sweep,
 * faceted by proposer, colored by search arm (blue one-shot / orange iterative).
 * Hard tasks use the deconfounded 4096-token reruns.
 *
 * Usage: npx tsx experiments/figures-e2.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const RESULTS = resolve(import.meta.dirname, "results");
const FIG_DIR = resolve(import.meta.dirname, "figures");

const C = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  oneShot: "#2a78d6",
  iterative: "#eb6834",
};
const FONT = `font-family="system-ui, -apple-system, 'Segoe UI', sans-serif"`;

interface RunResult {
  proposer: string;
  arm: string;
  exact: boolean | null;
  error: string | null;
}

const TASKS = [
  { key: "bounded-square", file: "summary.json", label: "bounded-square" },
  { key: "signed-window", file: "summary-signed-window-4k.json", label: "signed-window" },
  { key: "penalty-sum", file: "summary-penalty-sum-4k.json", label: "penalty-sum" },
];
const PANELS = [
  { proposer: "claude-sonnet-5", label: "Claude Sonnet 5" },
  { proposer: "claude-haiku-4-5", label: "Claude Haiku 4.5" },
];

function wilson(successes: number, n: number): { low: number; high: number } {
  if (n === 0) return { low: 0, high: 1 };
  const z = 1.959964;
  const p = successes / n;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const margin = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}

const cells = new Map<string, { exact: number; n: number }>();
for (const task of TASKS) {
  const runs = (JSON.parse(readFileSync(resolve(RESULTS, task.file), "utf8")) as { runs: RunResult[] }).runs;
  for (const run of runs.filter((entry) => entry.error === null && entry.exact !== null)) {
    const key = `${task.key}|${run.proposer}|${run.arm}`;
    const cell = cells.get(key) ?? { exact: 0, n: 0 };
    cell.n += 1;
    if (run.exact) cell.exact += 1;
    cells.set(key, cell);
  }
}

const width = 720;
const height = 400;
const margin = { top: 64, right: 16, bottom: 76, left: 56 };
const panelGap = 32;
const panelWidth = (width - margin.left - margin.right - panelGap) / PANELS.length;
const plotHeight = height - margin.top - margin.bottom;
const yOf = (rate: number) => margin.top + plotHeight * (1 - rate);
const barWidth = 20;

const parts: string[] = [];
parts.push(`<rect width="${width}" height="${height}" fill="${C.surface}"/>`);
parts.push(
  `<text x="${margin.left}" y="24" ${FONT} font-size="15" font-weight="600" fill="${C.ink}">Exact-solve rate across the task-difficulty sweep (budget: 4 proposal calls)</text>`,
);
parts.push(
  `<rect x="${margin.left}" y="${35}" width="12" height="12" rx="3" fill="${C.oneShot}"/>`,
  `<text x="${margin.left + 18}" y="${45}" ${FONT} font-size="12" fill="${C.ink2}">one-shot</text>`,
  `<rect x="${margin.left + 92}" y="${35}" width="12" height="12" rx="3" fill="${C.iterative}"/>`,
  `<text x="${margin.left + 110}" y="${45}" ${FONT} font-size="12" fill="${C.ink2}">iterative</text>`,
  `<text x="${width - margin.right}" y="${45}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">hard tasks: 4096-token reruns</text>`,
);

PANELS.forEach((panel, panelIndex) => {
  const x0 = margin.left + panelIndex * (panelWidth + panelGap);
  for (const tick of [0, 0.5, 1]) {
    const y = yOf(tick);
    parts.push(`<line x1="${x0}" y1="${y}" x2="${x0 + panelWidth}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`);
    if (panelIndex === 0) {
      parts.push(`<text x="${x0 - 8}" y="${y + 4}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">${tick * 100}%</text>`);
    }
  }
  parts.push(
    `<text x="${x0 + panelWidth / 2}" y="${height - 10}" ${FONT} font-size="12" fill="${C.ink2}" text-anchor="middle">${panel.label}</text>`,
  );
  TASKS.forEach((task, taskIndex) => {
    const groupCenter = x0 + (panelWidth * (taskIndex + 0.5)) / TASKS.length;
    parts.push(
      `<text x="${groupCenter}" y="${height - margin.bottom + 18}" ${FONT} font-size="10" fill="${C.muted}" text-anchor="middle">${task.label}</text>`,
    );
    (["one-shot", "iterative"] as const).forEach((arm, armIndex) => {
      const cell = cells.get(`${task.key}|${panel.proposer}|${arm}`);
      if (!cell) return;
      const rate = cell.exact / cell.n;
      const interval = wilson(cell.exact, cell.n);
      const x = groupCenter + (armIndex === 0 ? -barWidth - 1 : 1);
      const yTop = yOf(rate);
      const color = arm === "one-shot" ? C.oneShot : C.iterative;
      if (yOf(0) - yTop >= 4) {
        parts.push(
          `<path d="M ${x} ${yOf(0)} L ${x} ${yTop + 4} Q ${x} ${yTop} ${x + 4} ${yTop} L ${x + barWidth - 4} ${yTop} Q ${x + barWidth} ${yTop} ${x + barWidth} ${yTop + 4} L ${x + barWidth} ${yOf(0)} Z" fill="${color}"/>`,
        );
      } else if (yOf(0) - yTop > 0) {
        parts.push(`<rect x="${x}" y="${yTop}" width="${barWidth}" height="${yOf(0) - yTop}" fill="${color}"/>`);
      }
      const whiskerX = x + barWidth / 2;
      parts.push(
        `<line x1="${whiskerX}" y1="${yOf(interval.low)}" x2="${whiskerX}" y2="${yOf(interval.high)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
      );
      const anchor = armIndex === 0 ? "end" : "start";
      const labelX = armIndex === 0 ? whiskerX + 5 : whiskerX - 5;
      parts.push(
        `<text x="${labelX}" y="${yOf(interval.high) - 5}" ${FONT} font-size="10" fill="${C.ink}" text-anchor="${anchor}">${cell.exact}/${cell.n}</text>`,
      );
    });
  });
  parts.push(`<line x1="${x0}" y1="${yOf(0)}" x2="${x0 + panelWidth}" y2="${yOf(0)}" stroke="${C.baseline}" stroke-width="1"/>`);
});

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Exact-solve rates across three tasks of increasing difficulty, faceted by proposer">\n${parts.join("\n")}\n</svg>`;
mkdirSync(FIG_DIR, { recursive: true });
writeFileSync(resolve(FIG_DIR, "fig4-difficulty-sweep.svg"), svg);
console.log("[figures] wrote fig4-difficulty-sweep.svg");
