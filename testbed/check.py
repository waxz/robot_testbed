"""Regression check: run configs headless and evaluate their `checks:` blocks.

    python -m testbed.check                      # every config in configs/ that has checks
    python -m testbed.check configs/panda_diff_ik.yaml configs/go2_pd_stand.yaml
Exit code 0 when everything passes, 1 otherwise. Report: outputs/check_report.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="*")
    p.add_argument("--out", default=str(ROOT / "outputs/check_report.json"))
    args = p.parse_args()
    from testbed.checks import evaluate
    from testbed.config import load_config

    paths = [Path(c) for c in args.configs] or sorted((ROOT / "configs").glob("*.yaml"))
    report, all_ok = [], True
    for path in paths:
        cfg = load_config(path)
        checks = cfg.get("checks")
        if not checks:
            print(f"-- {path.stem}: no checks, skipped", flush=True)
            continue
        out = ROOT / "outputs" / f"check_{path.stem}.json"
        t0 = time.time()
        # one process per config: controllers may import heavy, process-global libraries
        proc = subprocess.run([sys.executable, "-m", "testbed.run", str(path), "--headless", "--out", str(out)],
                              cwd=ROOT, capture_output=True, text=True)
        wall = round(time.time() - t0, 1)
        if proc.returncode != 0 or not out.exists():
            all_ok = False
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            report.append(dict(config=path.stem, passed=False, wall_s=wall, error=tail))
            print(f"FAIL {path.stem}: run failed ({wall} s)\n     " + "\n     ".join(tail), flush=True)
            continue
        res = json.loads(out.read_text())
        results = evaluate(res, checks)
        ok = all(r["passed"] for r in results)
        all_ok &= ok
        report.append(dict(config=path.stem, passed=ok, wall_s=wall, checks=results))
        print(f"{'PASS' if ok else 'FAIL'} {path.stem} ({wall} s)", flush=True)
        for r in results:
            print(f"     {'ok ' if r['passed'] else 'BAD'} {r['check']:42s} {str(r['value']):>10s}  ({r['bound']})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(passed=all_ok, configs=report), indent=1))
    print(f"\n{'ALL PASSED' if all_ok else 'FAILURES'}: {sum(r['passed'] for r in report)}/{len(report)} configs  -> {args.out}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
