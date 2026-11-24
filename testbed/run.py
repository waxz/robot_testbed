"""Command line entry point.

    python -m testbed.run configs/forklift_pure_pursuit.yaml                 # web UI (port from config or --port)
    python -m testbed.run configs/panda_diff_ik.yaml --headless --seconds 10
    python -m testbed.run configs/g1_rl_joystick.yaml --set controller.backend=ort-cuda
"""
from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--out", default=None, help="headless: write results JSON here")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (YAML syntax)")
    args = p.parse_args()

    from testbed.config import load_config
    from testbed.loop import TestBed
    cfg = load_config(args.config, args.set)
    tb = TestBed(cfg)
    if args.headless:
        res = tb.run_headless(args.seconds, args.out)
        print("RESULT=" + json.dumps({k: v for k, v in res.items() if k != "trace"}, default=str), flush=True)
    else:
        from testbed.viewer import run_viewer
        run_viewer(tb, args.port or cfg.get("viewer", {}).get("port", 8090))


if __name__ == "__main__":
    main()
