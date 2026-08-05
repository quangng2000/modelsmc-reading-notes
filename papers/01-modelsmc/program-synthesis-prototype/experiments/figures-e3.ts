/**
 * Experiment-3 figure: the particles × iterations frontier at fixed budget 16
 * (Claude Haiku 4.5 on foldr-bounded-square). Single series — one variable
 * (refinement depth) sweeps left to right, so no legend is needed.
 *
 * Usage: npx tsx experiments/figures-e3.ts
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
  series: "#2a78d6",
};
const FONT = `font-family="system-ui, -apple-system, 'Segoe UI', sans-serif"`;

interface RunResult {
  arm: string;
  exact: boolean | null;
  bestLoss: number | null;
  error: string | null;
}

function wilson(successes: number, n: number): { low: number; high: number } {
  if (n === 0) return { low: 0, high: 1 };
  const z = 1.959964;
  const p = successes / n;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const margin = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return { low: Math.max(0, center - margin), high: Math.min(1, center + margin) };
}

const runs = (
  JSON.parse(readFileSync(resolve(RESULTS, "summary-budget16-haiku.json"), "utf8")) as {
    runs: RunResult[];
  }
).runs.filter((run) => run.error === null && run.exact !== null);

const CONFIGS = [
  { arm: "16x1", label: "16 × 1", sub: "pure sampling" },
  { arm: "8x2", label: "8 × 2", sub: "" },
  { arm: "4x4", label: "4 × 4", sub: "" },
  { arm: "2x8", label: "2 × 8", sub: "deep refinement" },
];

const width = 720;
const height = 400;
const margin = { top: 76, right: 16, bottom: 84, left: 56 };
const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;
const yOf = (rate: number) => margin.top + plotHeight * (1 - rate);
const barWidth = 24;

const parts: string[] = [];
parts.push(`<rect width="${width}" height="${height}" fill="${C.surface}"/>`);
parts.push(
  `<text x="${margin.left}" y="24" ${FONT} font-size="15" font-weight="600" fill="${C.ink}">The particles × iterations frontier at a fixed 16-call budget</text>`,
  `<text x="${margin.left}" y="44" ${FONT} font-size="12" fill="${C.ink2}">Claude Haiku 4.5 on foldr-bounded-square — exact-solve rate with 95% Wilson CI, 10 seeds per configuration</text>`,
);
for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
  const y = yOf(tick);
  parts.push(`<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`);
  parts.push(`<text x="${margin.left - 8}" y="${y + 4}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">${tick * 100}%</text>`);
}
CONFIGS.forEach((config, index) => {
  const armRuns = runs.filter((run) => run.arm === config.arm);
  const exact = armRuns.filter((run) => run.exact).length;
  const rate = exact / armRuns.length;
  const interval = wilson(exact, armRuns.length);
  const meanLoss = armRuns.reduce((sum, run) => sum + (run.bestLoss ?? 0), 0) / armRuns.length;
  const groupCenter = margin.left + (plotWidth * (index + 0.5)) / CONFIGS.length;
  const x = groupCenter - barWidth / 2;
  const yTop = yOf(rate);
  if (yOf(0) - yTop >= 4) {
    parts.push(
      `<path d="M ${x} ${yOf(0)} L ${x} ${yTop + 4} Q ${x} ${yTop} ${x + 4} ${yTop} L ${x + barWidth - 4} ${yTop} Q ${x + barWidth} ${yTop} ${x + barWidth} ${yTop + 4} L ${x + barWidth} ${yOf(0)} Z" fill="${C.series}"/>`,
    );
  } else if (yOf(0) - yTop > 0) {
    parts.push(`<rect x="${x}" y="${yTop}" width="${barWidth}" height="${yOf(0) - yTop}" fill="${C.series}"/>`);
  }
  const whiskerX = groupCenter;
  parts.push(
    `<line x1="${whiskerX}" y1="${yOf(interval.low)}" x2="${whiskerX}" y2="${yOf(interval.high)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
    `<line x1="${whiskerX - 4}" y1="${yOf(interval.low)}" x2="${whiskerX + 4}" y2="${yOf(interval.low)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
    `<line x1="${whiskerX - 4}" y1="${yOf(interval.high)}" x2="${whiskerX + 4}" y2="${yOf(interval.high)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
    `<text x="${whiskerX}" y="${yOf(interval.high) - 6}" ${FONT} font-size="11" fill="${C.ink}" text-anchor="middle">${exact}/${armRuns.length}</text>`,
    `<text x="${groupCenter}" y="${height - margin.bottom + 22}" ${FONT} font-size="12" fill="${C.ink2}" text-anchor="middle">${config.label}</text>`,
    `<text x="${groupCenter}" y="${height - margin.bottom + 38}" ${FONT} font-size="10" fill="${C.muted}" text-anchor="middle">mean loss ${meanLoss.toFixed(1)}</text>`,
  );
  if (config.sub) {
    parts.push(
      `<text x="${groupCenter}" y="${height - margin.bottom + 54}" ${FONT} font-size="10" fill="${C.muted}" text-anchor="middle">${config.sub}</text>`,
    );
  }
});
parts.push(
  `<line x1="${margin.left}" y1="${yOf(0)}" x2="${width - margin.right}" y2="${yOf(0)}" stroke="${C.baseline}" stroke-width="1"/>`,
  `<text x="${width / 2}" y="${height - 8}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="middle">particles × iterations (proposal budget fixed at 16 calls per run)</text>`,
);

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Bar chart of exact-solve rate across four particles-by-iterations configurations at a fixed 16-call budget">\n${parts.join("\n")}\n</svg>`;
mkdirSync(FIG_DIR, { recursive: true });
writeFileSync(resolve(FIG_DIR, "fig5-frontier.svg"), svg);
console.log("[figures] wrote fig5-frontier.svg");
