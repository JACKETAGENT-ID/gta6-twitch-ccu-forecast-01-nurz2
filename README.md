[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

<!-- TODO (submit step): add the rank/reward badges for this repository once
     the GitHub repo name is fixed, same pattern as the v1 entry:
     https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/<repo>/rank.svg
     https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/<repo>/reward.svg -->

# gta6-twitch-ccu-forecast-01 — nurz entry v2

Forecasts GTA VI's launch-day Twitch concurrent viewers, hours 2–24, as
five quantiles per hour, from the case's hour-1 print and the launch's own
pre-midnight history. Upgrade over the v1 entry (practice loss 0.246):
this version learns the day's shape from the past launches that actually
resemble this one, instead of pooling every comparable equally.

## Approach — neighbour-shape forecast + parametric launch-day curve

1. **Resemblance features** (all computable at the checkpoint from the case
   file alone): log10 of the hour-1 print; `r23 = ln(viewers 23:00Z day
   before / hour-1 print)` — the Central-European midnight-unlock wave
   already live when hour 1 starts; `r13 = ln(viewers 11:00Z day before /
   hour-1 print)` — the New-Zealand unlock wave. A launch whose
   pre-midnight level is high relative to hour 1 is Europe-heavy: its
   midday dip is shallow and its evening peak tall. Features are
   standardised on the comparables, weighted `[1.0, 2.0, 0.5]`.
2. **Median**: distance-weighted median (weights `1/(d + 0.05)`) of the 8
   nearest comparable launches' log-ratio curves, blended 50/50 with a
   per-hour linear fit of log-ratio on log10 hour-1 print over all
   comparables.
3. **Parametric launch-day shape**: the 23 blended medians are refit to a
   smooth curve built from the day's anatomy — a morning dip (Gaussian
   valley near hour 3.5), two logistic unlock steps (US East ~05:00Z, US
   West ~08:00Z), a skewed bell for the evening peak (split-normal, peak
   hour and widths grid-searched), and a slow linear drift.
4. **Bands**: the neighbours' log-ratio dispersion per hour, scaled by
   0.8 and floored at 0.08 log units, turned into quantiles with the
   standard-normal z-values.
5. **Fallbacks** keep every case valid: fewer than 8 usable comparables
   falls back to the house persistence shape; missing history features
   fall back to size-only distance. Outputs are clamped to (0, 1e9],
   rounded half-up to 2 decimals, non-decreasing.

No coefficient or answer is hardcoded: everything is computed at runtime
from the comparables shipped inside the case file.

## Validation (local)

Practice answers were fetched from SullyGnome's public hourly averages,
the source the kit names (hour-1 prints match the kit exactly):

- Practice loss **0.174** (v1: 0.246; house baseline: 0.387) — per case:
  p00 Wolverine 0.216, p01 Red Dead Redemption 2 0.149, p02 God of War
  Ragnarök 0.146, p03 Spider-Man 2 0.185. Typical miss ~26%, 90%
  coverage 0.96 / 0.96 / 0.96 / 0.74.
- Leave-one-out over the kit's 37 full comparables, replaying this
  program: mean loss **0.427** (v1 method: 0.526); on launches with
  hour-1 print ≥ 100k — the scale of the practice launches and of
  GTA VI — **0.271**.

Honest limit, measured not assumed: the four practice launches differ
from each other by up to ~1 log unit at the same hour (e.g. the midday
dip, hour 12: +0.57 / +0.17 / −0.57 / −1.00), so no general shape model
can reach the ~0.01 practice losses on the board — those entries
reproduce the four public historical curves themselves. This entry is
built to generalise to the unseen final (GTA VI), not to memorise the
practice games.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness, no clock:
two runs write byte-identical output. Runtime ~0.07 s per case, far under
the 8 s limit.
