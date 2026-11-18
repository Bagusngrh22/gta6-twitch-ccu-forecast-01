// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only, no network).
//
// Model: comparables-conditional log-ratio regression.
//
// The house persistence baseline forecasts every hour at hour 1's print. It
// fails badly on launches whose audience grows through the day, and the growth
// is strongly predictable: across the 38 comparable launches in a practice case,
// log(hour-1 viewers) correlates -0.94 with log(peak ratio). A launch that is
// small in hour 1 (late single global unlock, midnight in only a few time zones)
// grows many-fold; a launch already huge in hour 1 (rolling midnight, whole
// world awake) stays flat or declines. So, per forecast hour h = 2..24, fit an
// ordinary least-squares line of log(viewers_h / viewers_1) on log(viewers_1)
// across the comparables in the case file, and forecast the target from its own
// hour-1 print. The spread is the robust residual scale for that hour
// (1.4826 x median absolute residual, floored at 0.10, capped at 1.0), so the
// five quantiles are q_tau = p1 * exp(mu_h) * exp(z_tau * s_h) with the usual
// standard-normal z values. The fitted median is clamped to the range of
// log-ratios actually observed for that hour, so an out-of-range hour-1 print
// cannot extrapolate to an absurd ratio.
//
// Validation (local, against SullyGnome hourly averages pulled for the four
// practice launches' release days; hour-1 prints match the case files
// exactly: 23,357 / 116,550 / 269,517 / 119,087):
//   practice mean pinball loss 0.224 vs house persistence 0.387
//   (p00 0.203, p01 0.226, p02 0.233, p03 0.232)
//   leave-one-out over the 37 comparable launches with >=13 release-day hours:
//   mean 0.423 vs persistence 0.930, median 0.283 vs 0.325.
// Adding pre-launch peak or hour-1 channels as extra features did not help
// (LOO 0.450 / 0.427), so the model uses log(hour-1 viewers) only.
//
// Everything the program knows is in the case file it is given; it is fully
// deterministic (no randomness, no timestamps, no network) and runs in well
// under the 8,000 ms per-case cap.
//
// Input: every <id>.in in the input directory (the grader runs one case at a
// time, with only that case's file there).
// Output: <id>.out in the output directory:
//   {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, exactly 23 rows,
//   hours 2..24 each once, each q five finite numbers > 0 and at most 1e9,
//   non-decreasing, at most 65,536 bytes.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Minimum comparables needed to fit a line for an hour; below that, the median log-ratio. */
const MIN_FIT_POINTS = 8;
/** Fallback spread (log units) when too few comparables have an hour. */
const FALLBACK_SPREAD = 0.3;

interface CaseFile {
  id: string;
  known: Array<{ hour: number; viewers: number }>;
  forecast?: Array<{ hour: number }>;
  comparables?: Array<{
    day_start: string;
    hourly: Array<{ from: string; viewers: number }>;
  }>;
}

/** x rounded to 2 decimals (half up), matching the starter's rounding. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Median of a non-empty array of numbers. */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Release-day hour (1..24) -> viewers, from a comparable's hourly points. */
function releaseDayHours(comp: {
  day_start: string;
  hourly: Array<{ from: string; viewers: number }>;
}): Map<number, number> {
  const start = Date.parse(comp.day_start);
  const out = new Map<number, number>();
  if (!Number.isFinite(start)) return out;
  for (const p of comp.hourly ?? []) {
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && p.viewers > 0) out.set(h, p.viewers);
  }
  return out;
}

/** Ordinary least squares of y on x; returns {a, b} for y = a + b*x. */
function ols(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  const mx = xs.reduce((t, v) => t + v, 0) / n;
  const my = ys.reduce((t, v) => t + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

/** Forecast one case file. */
function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const known1 = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, known1?.viewers ?? 1);

  // Per-hour training points: (log hour-1 viewers, log ratio hour h / hour 1).
  const perHour = new Map<number, Array<{ x: number; y: number }>>();
  for (const comp of c.comparables ?? []) {
    const hrs = releaseDayHours(comp);
    const v1 = hrs.get(1);
    if (!v1) continue;
    const x = Math.log(v1);
    for (let h = 2; h <= 24; h++) {
      const vh = hrs.get(h);
      if (!vh) continue;
      const arr = perHour.get(h) ?? [];
      arr.push({ x, y: Math.log(vh / v1) });
      perHour.set(h, arr);
    }
  }

  const hoursToForecast = (c.forecast ?? []).map((f) => f.hour).filter((h) => h >= 2 && h <= 24);
  const hours = (hoursToForecast.length === 23 ? hoursToForecast : Array.from({ length: 23 }, (_, i) => i + 2)).sort((a, b) => a - b);

  const out = hours.map((h) => {
    const pts = perHour.get(h) ?? [];
    let mu: number;
    let spread: number;
    if (pts.length >= MIN_FIT_POINTS) {
      const fit = ols(pts.map((p) => p.x), pts.map((p) => p.y));
      mu = fit.a + fit.b * Math.log(p1);
      const resid = pts.map((p) => Math.abs(p.y - (fit.a + fit.b * p.x)));
      spread = Math.min(1.0, Math.max(0.1, 1.4826 * median(resid)));
      // Never extrapolate the median beyond the log-ratios actually observed.
      const ys = pts.map((p) => p.y);
      mu = Math.min(Math.max(mu, Math.min(...ys)), Math.max(...ys));
    } else if (pts.length > 0) {
      mu = median(pts.map((p) => p.y));
      spread = FALLBACK_SPREAD;
    } else {
      mu = 0;
      spread = FALLBACK_SPREAD;
    }
    const q50 = p1 * Math.exp(mu);
    const q: number[] = [];
    let prev = 0.01;
    for (const z of Z) {
      const v = Math.min(1e9, Math.max(0.01, round2(q50 * Math.exp(z * spread))));
      // Keep the row non-decreasing even after rounding/clamping.
      const clamped = Math.max(v, prev);
      q.push(clamped);
      prev = clamped;
    }
    return { hour: h, q };
  });

  return { hours: out };
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const file of files) {
    const id = file.slice(0, -3);
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
    const out = forecastCase(c);
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), JSON.stringify(out));
  }
}

main();
