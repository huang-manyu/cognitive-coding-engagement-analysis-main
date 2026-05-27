"""
Scan tasks/*/ssg/ssg_diagnosis.json (k_ssg_diagnosis output) for 22-lesson SSG metrics.

Writes:
  tasks/ssg_corpus_summary/ssg_per_lesson_metrics.csv — per lesson
  tasks/ssg_corpus_summary/ssg_corpus_aggregate.csv — corpus-wide stats & level counts

Logs in English.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from g_cluster_dynamics import TASK_DISPLAY_LABELS


def discover_task_ids(tasks_dir: Path) -> list[str]:
    out: list[str] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "ssg" / "ssg_diagnosis.json").is_file():
            out.append(d.name)
    return out


def load_lesson_row(task_id: str, path: Path) -> dict[str, Any] | None:
    try:
        diag = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skip {task_id}: failed to read JSON ({e})")
        return None

    meta = diag.get("meta") or {}
    eo = (diag.get("entropy") or {}).get("overall") or {}
    attractors = diag.get("attractors") or []

    states_joined = " | ".join(str(a.get("state", "")).strip() for a in attractors if a.get("state"))

    return {
        "task_id": task_id,
        "lesson_title": TASK_DISPLAY_LABELS.get(task_id, ""),
        "n_attractor_cells": len(attractors),
        "attractor_cells_states": states_joined,
        "n_attractor_regions": diag.get("attractor_region_component_count", ""),
        "visited_cells": meta.get("cell_range", ""),
        "total_visits": meta.get("total_visits", ""),
        "trajectory_events": meta.get("trajectory_points", ""),
        "dispersion": meta.get("dispersion", ""),
        "visited_entropy": eo.get("visited_entropy", ""),
        "duration_entropy": eo.get("duration_entropy", ""),
        "transition_entropy": eo.get("transition_entropy", ""),
        "visited_entropy_level": eo.get("visited_entropy_level", ""),
        "duration_entropy_level": eo.get("duration_entropy_level", ""),
        "transition_entropy_level": eo.get("transition_entropy_level", ""),
    }


def write_per_lesson_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "lesson_title",
        "n_attractor_cells",
        "attractor_cells_states",
        "n_attractor_regions",
        "visited_cells",
        "total_visits",
        "trajectory_events",
        "dispersion",
        "visited_entropy",
        "duration_entropy",
        "transition_entropy",
        "visited_entropy_level",
        "duration_entropy_level",
        "transition_entropy_level",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """One row per summary line: stat_name, task_id_if_any, value, detail."""
    lines: list[dict[str, str]] = []

    def num(r: dict[str, Any], k: str) -> float:
        v = r.get(k)
        if v is None or v == "":
            return float("nan")
        return float(v)

    vc = [(r["task_id"], num(r, "visited_cells")) for r in rows]
    vc = [(t, v) for t, v in vc if v == v]  # drop nan
    if vc:
        mx = max(vc, key=lambda x: x[1])
        mn = min(vc, key=lambda x: x[1])
        avg = statistics.mean(v[1] for v in vc)
        lines.append(
            {
                "stat_name": "visited_cells_max",
                "task_id": mx[0],
                "value": str(int(mx[1]) if mx[1] == int(mx[1]) else round(mx[1], 4)),
                "detail": TASK_DISPLAY_LABELS.get(mx[0], ""),
            }
        )
        lines.append(
            {
                "stat_name": "visited_cells_min",
                "task_id": mn[0],
                "value": str(int(mn[1]) if mn[1] == int(mn[1]) else round(mn[1], 4)),
                "detail": TASK_DISPLAY_LABELS.get(mn[0], ""),
            }
        )
        lines.append({"stat_name": "visited_cells_mean", "task_id": "", "value": f"{avg:.6f}", "detail": ""})

    disp = [(r["task_id"], num(r, "dispersion")) for r in rows]
    disp = [(t, v) for t, v in disp if v == v]
    if disp:
        mx = max(disp, key=lambda x: x[1])
        mn = min(disp, key=lambda x: x[1])
        lines.append(
            {
                "stat_name": "dispersion_max",
                "task_id": mx[0],
                "value": f"{mx[1]:.6f}",
                "detail": TASK_DISPLAY_LABELS.get(mx[0], ""),
            }
        )
        lines.append(
            {
                "stat_name": "dispersion_min",
                "task_id": mn[0],
                "value": f"{mn[1]:.6f}",
                "detail": TASK_DISPLAY_LABELS.get(mn[0], ""),
            }
        )

    n = len(rows)
    lines.append({"stat_name": "n_lessons_in_corpus", "task_id": "", "value": str(n), "detail": ""})

    for level_key in (
        "transition_entropy_level",
        "visited_entropy_level",
        "duration_entropy_level",
    ):
        ctr: Counter[str] = Counter()
        for r in rows:
            lv = str(r.get(level_key, "")).strip()
            if lv in ("高", "中", "低"):
                ctr[lv] += 1
        for bucket in ("高", "中", "低"):
            cnt = ctr.get(bucket, 0)
            tids = [r["task_id"] for r in rows if str(r.get(level_key, "")).strip() == bucket]
            lines.append(
                {
                    "stat_name": f"{level_key}_{bucket}_count",
                    "task_id": "",
                    "value": str(cnt),
                    "detail": ";".join(tids),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["stat_name", "task_id", "value", "detail"])
        w.writeheader()
        for line in lines:
            w.writerow(line)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="SSG corpus summary from ssg_diagnosis.json files.")
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=root / "tasks",
        help="Tasks root (default: repo tasks/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "tasks" / "ssg_corpus_summary",
        help="Output folder (default: tasks/ssg_corpus_summary)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks_dir: Path = args.tasks_dir
    if not tasks_dir.is_dir():
        print(f"Not a directory: {tasks_dir}")
        sys.exit(1)

    ids = discover_task_ids(tasks_dir)
    if not ids:
        print("No tasks with ssg/ssg_diagnosis.json")
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for tid in ids:
        p = tasks_dir / tid / "ssg" / "ssg_diagnosis.json"
        row = load_lesson_row(tid, p)
        if row:
            rows.append(row)

    out_dir = args.out_dir
    per_csv = out_dir / "ssg_per_lesson_metrics.csv"
    agg_csv = out_dir / "ssg_corpus_aggregate.csv"
    write_per_lesson_csv(per_csv, rows)
    write_aggregate_csv(agg_csv, rows)

    print(f"Wrote {per_csv} ({len(rows)} lessons)")
    print(f"Wrote {agg_csv}")


if __name__ == "__main__":
    main()
