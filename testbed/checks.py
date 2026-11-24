"""Pass/fail checks on headless results, declared per config:

    checks:
      - {metric: falls, max: 0}
      - {metric: final_state.pallet.pos.2, min: 0.2}                 # dotted path into the result
      - {metric: trace.cross_track_m, agg: mean, after: 8, max: 0.35} # over the 0.5 s status trace
      - {metric: trace.pos_err_cm, agg: max, at: [2.5, 4.5], max: 0.5}  # trace samples nearest these times
      - {metric: timing_ms.realtime_factor, min: 1.0}

agg: mean | max | min | last (default last); after / before: time window on the trace; at: sample times.
"""
from __future__ import annotations

import numpy as np


def _dig(obj, path: str):
    for k in path.split("."):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


def metric_value(res: dict, chk: dict) -> float:
    m = chk["metric"]
    if not m.startswith("trace."):
        return float(_dig(res, m))
    field = m[len("trace."):]
    rows = [r for r in res["trace"] if field in r]
    if "at" in chk:
        ts = np.array([r["t"] for r in rows])
        rows = [rows[int(np.argmin(np.abs(ts - t)))] for t in chk["at"]]
    rows = [r for r in rows if chk.get("after", -np.inf) <= r["t"] <= chk.get("before", np.inf)]
    vals = np.array([float(r[field]) for r in rows])
    if len(vals) == 0:
        raise ValueError(f"no trace samples for {m}")
    return float({"mean": vals.mean, "max": vals.max, "min": vals.min, "last": lambda: vals[-1]}[chk.get("agg", "last")]())


def evaluate(res: dict, checks: list[dict]) -> list[dict]:
    out = []
    for chk in checks:
        desc = chk["metric"] + (f" [{chk['agg']}]" if "agg" in chk else "")
        try:
            v = metric_value(res, chk)
            ok = (v >= chk["min"] if "min" in chk else True) and (v <= chk["max"] if "max" in chk else True)
            bound = " ".join(f"{k} {chk[k]}" for k in ("min", "max") if k in chk)
            out.append(dict(check=desc, value=round(v, 4), bound=bound, passed=bool(ok)))
        except Exception as e:  # a missing metric is a failure, not a crash
            out.append(dict(check=desc, value=None, bound=str(e)[:80], passed=False))
    return out
