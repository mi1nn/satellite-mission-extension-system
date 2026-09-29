#!/usr/bin/env python3
"""Compare docking controllers from `vision_capture.py` result JSON files.

Run the same start twice, once per `docking_control.mode`, then compare:

    ~/isaac-sim/python.sh scripts/vision_capture.py --scenario dynamic --dock_only --headless --tag legacy
    ~/isaac-sim/python.sh scripts/vision_capture.py --scenario dynamic --dock_only --headless --tag coupled \\
        --set docking_control.mode=coupled_predictive
    python3 scripts/compare_docking_control.py logs/vision_capture/legacy_result.json logs/vision_capture/coupled_result.json

Several files of the same mode are aggregated (success rate, mean of the rest).
Pure standard library: no Isaac Sim needed.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# (key in metrics.docking_control, label, scale, unit)
ROWS = (
    ("docking_success", "Docking success rate", 100.0, "%"),
    ("docking_time_s", "Docking time", 1.0, "s"),
    ("probe_tip_position_error_at_dock_m", "Probe tip position error at dock", 1000.0, "mm"),
    ("probe_tip_orientation_error_at_dock_deg", "Probe tip orientation error at dock", 1.0, "deg"),
    ("relative_velocity_at_dock_mps", "Relative velocity at dock", 1000.0, "mm/s"),
    ("max_position_error_m", "Max lateral error (align/approach)", 1000.0, "mm"),
    ("max_orientation_error_deg", "Max orientation error (align/approach)", 1.0, "deg"),
    ("rollback_count", "Rollback count", 1.0, ""),
    ("emergency_stop_count", "Emergency stop count", 1.0, ""),
    ("alignment_state_switching_count", "Align <-> approach switches", 1.0, ""),
    ("max_contact_velocity_mps", "Max contact (in-nozzle) velocity", 1000.0, "mm/s"),
)


def load(paths):
    runs = defaultdict(list)
    for p in paths:
        data = json.loads(Path(p).read_text())
        dc = data.get("metrics", {}).get("docking_control")
        if dc is None:
            print(f"skip {p}: no metrics.docking_control (run predates the comparison logging?)")
            continue
        runs[dc.get("mode", "?")].append(dc)
    return runs


def aggregate(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="vision_capture result JSON files")
    args = parser.parse_args()
    runs = load(args.results)
    if not runs:
        raise SystemExit("no docking_control metrics found")
    modes = sorted(runs)
    width = max(len(r[1]) for r in ROWS) + 2
    print(f"{'metric':<{width}}" + "".join(f"{m + f' (n={len(runs[m])})':>28}" for m in modes))
    for key, label, scale, unit in ROWS:
        cells = []
        for m in modes:
            v = aggregate(r.get(key) for r in runs[m])
            cells.append("n/a" if v is None else f"{v * scale:.3f} {unit}".strip())
        print(f"{label:<{width}}" + "".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
