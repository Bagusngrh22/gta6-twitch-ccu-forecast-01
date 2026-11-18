# gta6-twitch-ccu-forecast-01

Forecast GTA VI's launch-day Twitch viewers hour by hour. A Node 22 program
(standard library only) reads one case file — hour 1's print plus 38
comparable past game launches — and writes five quantiles per hour for
hours 2..24 of 19 November 2026 (UTC).

## Model

Comparables-conditional log-ratio regression. The house persistence baseline
forecasts every hour at hour 1's print, but a launch's growth through the day
is strongly predictable from how big hour 1 was: across the comparables,
log(hour-1 viewers) correlates −0.94 with log(peak ratio). Small hour-1
launches grow many-fold; huge ones stay flat. So per hour, an OLS line of
log(viewers_h / viewers_1) on log(viewers_1) is fit across the comparables in
the case file, and the target is forecast from its own hour-1 print. The
spread is the robust residual scale for that hour (1.4826 × median absolute
residual, floored 0.10, capped 1.0). The fitted median is clamped to the
log-ratios actually observed for that hour.

## Validation (local, SullyGnome hourly averages for the four practice launches)

- Practice mean pinball loss **0.224** vs house persistence **0.387**
  (p00 0.203, p01 0.226, p02 0.233, p03 0.232)
- Leave-one-out over the 37 comparable launches with ≥13 release-day hours:
  mean **0.423** vs persistence **0.930**, median 0.283 vs 0.325

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

No network, no clock, no randomness: two runs write byte-identical outputs,
~68 ms per case (cap 8,000 ms), each output ~1.5 KB and valid
(23 rows, hours 2..24, quantiles > 0, ≤ 1e9, non-decreasing).

Entered on [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) —
$250, 20 paid places; a prize needs a loss strictly below the best house entry.
