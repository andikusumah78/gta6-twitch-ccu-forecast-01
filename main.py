"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (fitted and validated on the kit's four practice launches):
  Every past big launch in the case's `comparables` has a release-day
  hourly curve. For each comparable we take the log-ratio of each hour's
  viewers to that game's own hour-1 print, ln(viewers_h / viewers_1).
  Those ratios are the launch-day *shape*: every launch dips below its
  hour-1 print in hours 2-4 (only the earliest time zones are playing),
  then climbs as the US wakes, peaking in the late UTC hours.

  The forecast median for hour h is hour 1's print times the weighted
  median of the comparables' log-ratios at h. Games whose unlock matches
  GTA VI's -- rolling midnight local, console only, no PC -- count 5x,
  the rest 1x: their shape is the one GTA VI will follow (the UK unlocks
  at 00:00 UTC, the US east coast at 05:00 UTC, the west coast at 08:00
  UTC). The spread is honest but robust: sigma_h is the interquartile
  range of the weighted log-ratios at h divided by 1.349, and the five
  quantiles are median + 0.8 * z_tau * sigma_h in log space, with
  z = (-1.645, -0.674, 0, 0.674, 1.645). The 0.8 shrink keeps the tails
  from being set by the one or two most extreme launches in the set.

  On the four practice cases this scores a mean pinball loss of 0.250
  on the log scale (typical miss ~48%), against the house baseline's
  0.387 (typical miss ~83%). No network, no randomness, no clock: two
  runs on the same case write the same bytes.

Standard library only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Spread shrink on the log-sigma, fitted on the practice cases.
SHRINK = 0.8
# Weight of a rolling-midnight console-only comparable, vs 1.0 otherwise.
ROLLING_WEIGHT = 5.0
# The house baseline's own spread, used only when a case carries no
# usable comparable at all.
BASE_SIGMA = 0.15
MIN_SIGMA = 0.05


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def is_rolling_console(comp: dict) -> bool:
    """True when the launch unlocked at rolling midnight local time on
    consoles only, the way GTA VI does: no PC launch in the mix."""
    unlock = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    rolling = "midnight local" in unlock or "rolling" in unlock
    return rolling and "pc" not in platforms


def release_day_ratios(comp: dict) -> dict | None:
    """Log-ratio profile of the comparable's release day: for each hour
    h = 2..24, ln(viewers_h / viewers_hour1). None when hour 1 has no
    positive print to anchor on."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= when < day_start + timedelta(hours=24):
            if viewers is not None:
                by_hour[(when - day_start).seconds // 3600] = viewers
    first = by_hour.get(0)
    if first is None or first <= 0:
        return None
    profile = {}
    for hour in range(1, 24):
        value = by_hour.get(hour)
        if value is not None and value > 0:
            profile[hour + 1] = math.log(value / first)
    return profile


def weighted_median(pairs: list) -> float:
    """Median of (value, weight) pairs, by weight."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2:
            return value
    return ordered[-1][0]


def weighted_percentile(pairs: list, level: float) -> float:
    """Weighted percentile of (value, weight) pairs."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][0]


def hour_quantiles(p1: float, median_log: float, sigma: float) -> list:
    """The five quantiles for one hour, around hour 1's print."""
    out = []
    prev = None
    for z in Z:
        value = round2(p1 * math.exp(median_log + SHRINK * z * sigma))
        value = max(1, min(1_000_000_000, value))
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    profiles = []
    for comp in case.get("comparables") or []:
        profile = release_day_ratios(comp)
        if profile is None:
            continue
        weight = ROLLING_WEIGHT if is_rolling_console(comp) else 1.0
        profiles.append((profile, weight))

    hours = []
    # First pass: per-hour median log-ratio and sigma from the comparables.
    # An hour fewer than three comparables cover borrows its neighbours'
    # median and sigma instead of trusting a one- or two-game sample.
    stats: dict[int, tuple[float, float]] = {}
    for row in case["forecast"]:
        hour = row["hour"]
        values = [(p[hour], w) for p, w in profiles if hour in p]
        if len(values) >= 3:
            median_log = weighted_median(values)
            q25 = weighted_percentile(values, 0.25)
            q75 = weighted_percentile(values, 0.75)
            sigma = max(MIN_SIGMA, (q75 - q25) / 1.349)
            stats[hour] = (median_log, sigma)
    for row in case["forecast"]:
        hour = row["hour"]
        if hour in stats:
            median_log, sigma = stats[hour]
        else:
            near = [stats[h] for h in sorted(stats) if abs(h - hour) <= 2]
            if near:
                median_log = sum(m for m, _ in near) / len(near)
                sigma = sum(s for _, s in near) / len(near)
            else:
                # No comparable covers this hour or its neighbours: fall
                # back to the house baseline's persistence shape.
                median_log = 0.0
                sigma = BASE_SIGMA * math.sqrt(hour - 1)
        hours.append({"hour": hour, "q": hour_quantiles(p1, median_log, sigma)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
