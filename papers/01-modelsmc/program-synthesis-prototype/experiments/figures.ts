/**
 * Paper figures for the foldr-bounded-square matrix.
 *
 * Reads experiments/results/{summary,stats}.json and writes static SVGs to
 * experiments/figures/. Figures bake in the light chart surface so they stay
 * legible on both GitHub themes. Color encodes the search arm everywhere:
 * blue = one-shot, orange = iterative.
 *
 * Usage: npx tsx experiments/figures.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const RESULTS = resolve(import.meta.dirname, "results");
const FIG_DIR = resolve(import.meta.dirname, "figures");

// Reference palette (validated): series-1 blue, series-2 orange; ink/chrome tokens.
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

interface CellStats {
  proposer: string;
  arm: string;
  n: number;
  exactCount: number;
  successRate: number;
  wilsonLow: number;
  wilsonHigh: number;
  bestLosses: number[];
}
interface RunResult {
  proposer: string;
  arm: string;
  seed: number;
  exact: boolean | null;
  bestLoss: number | null;
  lineageLosses: number[];
  error: string | null;
}

const stats = JSON.parse(readFileSync(resolve(RESULTS, "stats.json"), "utf8")) as {
  cells: CellStats[];
};
const summary = JSON.parse(readFileSync(resolve(RESULTS, "summary.json"), "utf8")) as {
  runs: RunResult[];
};
const PROPOSERS = ["claude-sonnet-5", "claude-haiku-4-5", "qwen3-coder-30b", "catalog"];
const LABELS: Record<string, string> = {
  "claude-sonnet-5": "Claude Sonnet 5",
  "claude-haiku-4-5": "Claude Haiku 4.5",
  "qwen3-coder-30b": "Qwen3-Coder 30B",
  catalog: "Catalog (offline)",
};

function cell(proposer: string, arm: string): CellStats | undefined {
  return stats.cells.find((entry) => entry.proposer === proposer && entry.arm === arm);
}

function legend(x: number, y: number): string {
  return [
    `<rect x="${x}" y="${y - 9}" width="12" height="12" rx="3" fill="${C.oneShot}"/>`,
    `<text x="${x + 18}" y="${y + 1}" ${FONT} font-size="12" fill="${C.ink2}">one-shot (4 particles × 1 iteration)</text>`,
    `<rect x="${x + 232}" y="${y - 9}" width="12" height="12" rx="3" fill="${C.iterative}"/>`,
    `<text x="${x + 250}" y="${y + 1}" ${FONT} font-size="12" fill="${C.ink2}">iterative (2 particles × 2 iterations)</text>`,
  ].join("\n");
}

/* ---------------- Figure 1: success rate ---------------- */
function figureSuccessRate(): string {
  const width = 720;
  const height = 400;
  const margin = { top: 64, right: 16, bottom: 64, left: 56 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yOf = (rate: number) => margin.top + plotHeight * (1 - rate);
  const parts: string[] = [];
  parts.push(`<rect width="${width}" height="${height}" fill="${C.surface}"/>`);
  parts.push(
    `<text x="${margin.left}" y="24" ${FONT} font-size="15" font-weight="600" fill="${C.ink}">Exact-solution rate on foldr-bounded-square (budget: 4 proposal calls)</text>`,
  );
  parts.push(legend(margin.left, 44));
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yOf(tick);
    parts.push(`<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`);
    parts.push(`<text x="${margin.left - 8}" y="${y + 4}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">${tick * 100}%</text>`);
  }
  const groupWidth = plotWidth / PROPOSERS.length;
  const barWidth = 24;
  PROPOSERS.forEach((proposer, groupIndex) => {
    const groupCenter = margin.left + groupWidth * (groupIndex + 0.5);
    parts.push(
      `<text x="${groupCenter}" y="${height - margin.bottom + 22}" ${FONT} font-size="12" fill="${C.ink2}" text-anchor="middle">${LABELS[proposer]}</text>`,
    );
    (["one-shot", "iterative"] as const).forEach((arm, armIndex) => {
      const entry = cell(proposer, arm);
      if (!entry) return;
      // 2px surface gap between adjacent bars.
      const x = groupCenter + (armIndex === 0 ? -barWidth - 1 : 1);
      const yTop = yOf(entry.successRate);
      const barHeight = Math.max(0, yOf(0) - yTop);
      const color = arm === "one-shot" ? C.oneShot : C.iterative;
      if (barHeight >= 4) {
        // Rounded 4px data-end, square at the baseline.
        parts.push(
          `<path d="M ${x} ${yOf(0)} L ${x} ${yTop + 4} Q ${x} ${yTop} ${x + 4} ${yTop} L ${x + barWidth - 4} ${yTop} Q ${x + barWidth} ${yTop} ${x + barWidth} ${yTop + 4} L ${x + barWidth} ${yOf(0)} Z" fill="${color}"/>`,
        );
      } else if (barHeight > 0) {
        parts.push(`<rect x="${x}" y="${yTop}" width="${barWidth}" height="${barHeight}" fill="${color}"/>`);
      }
      // Wilson 95% CI whisker.
      const whiskerX = x + barWidth / 2;
      parts.push(
        `<line x1="${whiskerX}" y1="${yOf(entry.wilsonLow)}" x2="${whiskerX}" y2="${yOf(entry.wilsonHigh)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
        `<line x1="${whiskerX - 4}" y1="${yOf(entry.wilsonLow)}" x2="${whiskerX + 4}" y2="${yOf(entry.wilsonLow)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
        `<line x1="${whiskerX - 4}" y1="${yOf(entry.wilsonHigh)}" x2="${whiskerX + 4}" y2="${yOf(entry.wilsonHigh)}" stroke="${C.ink2}" stroke-width="1.5"/>`,
      );
      // Count label above the whisker, in ink (never the series color); the two
      // arms' labels splay outward so adjacent tall bars cannot collide.
      const labelAnchor = armIndex === 0 ? "end" : "start";
      const labelX = armIndex === 0 ? whiskerX + 6 : whiskerX - 6;
      parts.push(
        `<text x="${labelX}" y="${yOf(entry.wilsonHigh) - 6}" ${FONT} font-size="11" fill="${C.ink}" text-anchor="${labelAnchor}">${entry.exactCount}/${entry.n}</text>`,
      );
    });
  });
  parts.push(`<line x1="${margin.left}" y1="${yOf(0)}" x2="${width - margin.right}" y2="${yOf(0)}" stroke="${C.baseline}" stroke-width="1"/>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Grouped bar chart of exact-solution rates by proposer and search arm">\n${parts.join("\n")}\n</svg>`;
}

/* ---------------- Figure 2: final-loss strip plot ---------------- */
function figureLossStrip(): string {
  const width = 720;
  const height = 380;
  const margin = { top: 64, right: 16, bottom: 64, left: 56 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const runs = summary.runs.filter((run) => run.error === null && run.bestLoss !== null);
  const maxLoss = Math.max(24, ...runs.map((run) => run.bestLoss ?? 0));
  const yOf = (loss: number) => margin.top + plotHeight * (loss / maxLoss);
  const parts: string[] = [];
  parts.push(`<rect width="${width}" height="${height}" fill="${C.surface}"/>`);
  parts.push(
    `<text x="${margin.left}" y="24" ${FONT} font-size="15" font-weight="600" fill="${C.ink}">Final best-so-far loss per run (0 = exact solution; lower is better)</text>`,
  );
  parts.push(legend(margin.left, 44));
  const step = maxLoss <= 24 ? 4 : 5;
  for (let tick = 0; tick <= maxLoss; tick += step) {
    const y = yOf(tick);
    parts.push(`<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`);
    parts.push(`<text x="${margin.left - 8}" y="${y + 4}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">${tick}</text>`);
  }
  parts.push(
    `<text x="${width - margin.right}" y="${yOf(0) - 6}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">exact</text>`,
  );
  const groupWidth = plotWidth / PROPOSERS.length;
  PROPOSERS.forEach((proposer, groupIndex) => {
    const groupCenter = margin.left + groupWidth * (groupIndex + 0.5);
    parts.push(
      `<text x="${groupCenter}" y="${height - margin.bottom + 22}" ${FONT} font-size="12" fill="${C.ink2}" text-anchor="middle">${LABELS[proposer]}</text>`,
    );
    (["one-shot", "iterative"] as const).forEach((arm, armIndex) => {
      const armCenter = groupCenter + (armIndex === 0 ? -groupWidth / 5 : groupWidth / 5);
      const color = arm === "one-shot" ? C.oneShot : C.iterative;
      const armRuns = runs
        .filter((run) => run.proposer === proposer && run.arm === arm)
        .sort((left, right) => left.seed - right.seed);
      armRuns.forEach((run) => {
        // Deterministic seed-indexed jitter; (seed*3) mod 10 is distinct for seeds 1..10,
        // so no two runs in an arm share an x position.
        const jitter = ((run.seed * 3) % 10) * 3 - 13.5;
        parts.push(
          `<circle cx="${armCenter + jitter}" cy="${yOf(run.bestLoss!)}" r="5" fill="${color}" stroke="${C.surface}" stroke-width="2"/>`,
        );
      });
    });
  });
  parts.push(`<line x1="${margin.left}" y1="${yOf(0)}" x2="${width - margin.right}" y2="${yOf(0)}" stroke="${C.baseline}" stroke-width="1"/>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Strip plot of final losses per run by proposer and search arm">\n${parts.join("\n")}\n</svg>`;
}

/* ---------------- Figure 3: iterative lineage trajectories ---------------- */
function figureLineages(): string {
  const panels = PROPOSERS.filter((proposer) => proposer !== "catalog");
  const width = 720;
  const height = 316;
  const margin = { top: 72, right: 16, bottom: 48, left: 56 };
  const panelGap = 24;
  const panelWidth = (width - margin.left - margin.right - panelGap * (panels.length - 1)) / panels.length;
  const plotHeight = height - margin.top - margin.bottom;
  const runs = summary.runs.filter(
    (run) => run.error === null && run.arm === "iterative" && run.lineageLosses.length > 0,
  );
  // Round the domain up to a multiple of 8 so the last gridline always bounds the
  // data (a lineage can worsen past the seed loss before recovering).
  const maxLoss =
    Math.ceil(Math.max(24, ...runs.flatMap((run) => run.lineageLosses)) / 8) * 8;
  const maxSteps = Math.max(...runs.map((run) => run.lineageLosses.length));
  const yOf = (loss: number) => margin.top + plotHeight * (loss / maxLoss);
  const parts: string[] = [];
  parts.push(`<rect width="${width}" height="${height}" fill="${C.surface}"/>`);
  parts.push(
    `<text x="${margin.left}" y="24" ${FONT} font-size="15" font-weight="600" fill="${C.ink}">Champion-lineage loss along SMC ancestry (iterative arm, every seed)</text>`,
  );
  parts.push(
    `<text x="${margin.left}" y="42" ${FONT} font-size="12" fill="${C.ink2}">Each line is one run's winning ancestry: seed program → iteration-1 revision → iteration-2 revision</text>`,
  );
  panels.forEach((proposer, panelIndex) => {
    const x0 = margin.left + panelIndex * (panelWidth + panelGap);
    const xOf = (lineageStep: number) => x0 + (panelWidth * lineageStep) / Math.max(1, maxSteps - 1);
    parts.push(
      `<text x="${x0 + panelWidth / 2}" y="${height - 8}" ${FONT} font-size="12" fill="${C.ink2}" text-anchor="middle">${LABELS[proposer]}</text>`,
    );
    for (let tick = 0; tick <= maxLoss; tick += 8) {
      const y = yOf(tick);
      parts.push(`<line x1="${x0}" y1="${y}" x2="${x0 + panelWidth}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`);
      if (panelIndex === 0) {
        parts.push(`<text x="${x0 - 8}" y="${y + 4}" ${FONT} font-size="11" fill="${C.muted}" text-anchor="end">${tick}</text>`);
      }
    }
    // Step tick labels sit at the panel bottom; first/last anchor inward so
    // neighboring panels' labels can never collide.
    const tickY = margin.top + plotHeight + 14;
    for (let lineageStep = 0; lineageStep < maxSteps; lineageStep += 1) {
      const anchor = lineageStep === 0 ? "start" : lineageStep === maxSteps - 1 ? "end" : "middle";
      parts.push(
        `<text x="${xOf(lineageStep)}" y="${tickY}" ${FONT} font-size="10" fill="${C.muted}" text-anchor="${anchor}">${lineageStep === 0 ? "seed" : `iter ${lineageStep}`}</text>`,
      );
    }
    const panelRuns = runs.filter((run) => run.proposer === proposer);
    panelRuns.forEach((run, runIndex) => {
      // A length-1 lineage (champion = seed) has no line; jitter its lone dot so
      // identical runs read as a cluster instead of one stacked point.
      const loneJitter = run.lineageLosses.length === 1 ? runIndex * 10 : 0;
      const points = run.lineageLosses.map((loss, lineageStep) => `${xOf(lineageStep) + loneJitter},${yOf(loss)}`);
      if (points.length > 1) {
        parts.push(
          `<polyline points="${points.join(" ")}" fill="none" stroke="${C.iterative}" stroke-width="2" stroke-opacity="0.45" stroke-linejoin="round" stroke-linecap="round"/>`,
        );
      }
      const lastStep = run.lineageLosses.length - 1;
      parts.push(
        `<circle cx="${xOf(lastStep) + loneJitter}" cy="${yOf(run.lineageLosses[lastStep]!)}" r="4" fill="${C.iterative}" stroke="${C.surface}" stroke-width="2"/>`,
      );
    });
    const exactCount = panelRuns.filter((run) => run.lineageLosses[run.lineageLosses.length - 1] === 0).length;
    parts.push(
      `<text x="${x0 + panelWidth}" y="${yOf(0) - 8}" ${FONT} font-size="11" fill="${C.ink}" text-anchor="end">${exactCount}/${panelRuns.length} reach 0</text>`,
    );
    parts.push(`<line x1="${x0}" y1="${yOf(0)}" x2="${x0 + panelWidth}" y2="${yOf(0)}" stroke="${C.baseline}" stroke-width="1"/>`);
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Small-multiple line charts of champion lineage losses for the iterative arm">\n${parts.join("\n")}\n</svg>`;
}

mkdirSync(FIG_DIR, { recursive: true });
writeFileSync(resolve(FIG_DIR, "fig1-success-rate.svg"), figureSuccessRate());
writeFileSync(resolve(FIG_DIR, "fig2-final-loss.svg"), figureLossStrip());
writeFileSync(resolve(FIG_DIR, "fig3-lineage.svg"), figureLineages());
console.log(`[figures] wrote 3 figures to ${FIG_DIR}`);
