"""
Export one long-form CSV covering sudden jumps, plateaus, strong/weak attractor spans,
and phase-transition segments (A/B/C) for every task with ssg/ssg_diagnosis.json.

Primary inputs per task:
  - tasks/{id}/ssg/ssg_diagnosis.json  (critical_points, sliding_ssg_bridge, sliding_insights)
  - tasks/{id}/group/full.json + tasks/{id}/class/full.json  (trajectory for student-side stats)

Teacher-side span stats match k_ssg_diagnosis sliding_ssg_bridge when the same seq is used;
student-side stats are computed here (not stored in diagnosis JSON).

Output default: tasks/ssg_corpus_summary/ssg_lesson_span_detail.csv

Logs in English.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from g_cluster_dynamics import TASK_DISPLAY_LABELS
from k_ssg_diagnosis import (
    analyze_minute_span_teacher_only,
    load_group_bounds,
    normalize_teacher_result,
)


def discover_task_ids(tasks_dir: Path) -> list[str]:
    out: list[str] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "ssg" / "ssg_diagnosis.json").is_file():
            out.append(d.name)
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_seq_and_minute_map(task_dir: Path) -> tuple[list[tuple[int, str, str]], dict[int, int]] | None:
    class_path = task_dir / "class" / "full.json"
    group_path = task_dir / "group" / "full.json"
    if not class_path.is_file() or not group_path.is_file():
        return None
    try:
        class_data = load_json(class_path)
        group_data = load_json(group_path)
    except (OSError, json.JSONDecodeError):
        return None
    bounds, _ = load_group_bounds(group_data)
    results = class_data.get("results", [])
    seq: list[tuple[int, str, str]] = []
    for r in results:
        try:
            gid = int(r["group"])
        except (KeyError, TypeError, ValueError):
            continue
        if gid not in bounds:
            continue
        s = r["student"]["result"]
        t = normalize_teacher_result(r["teacher"]["result"])
        seq.append((gid, s, t))
    seq.sort(key=lambda x: x[0])
    if not seq:
        return None
    group_to_minute: dict[int, int] = {}
    for gid, (t0, t1) in bounds.items():
        mid = (t0 + t1) / 2.0
        group_to_minute[gid] = int(mid // 60) + 1
    return seq, group_to_minute


def analyze_minute_span_student_only(
    seq: list[tuple[int, str, str]],
    group_to_minute: dict[int, int],
    start_minute: int,
    end_minute: int,
) -> dict[str, Any]:
    """Marginal student strategy + student-strategy change edges within [start_minute, end_minute]."""
    events_in = [
        (gid, s, t)
        for gid, s, t in seq
        if start_minute <= group_to_minute.get(gid, -10**9) <= end_minute
    ]
    n_events = len(events_in)
    if n_events == 0:
        return {
            "n_events": 0,
            "unique_students": 0,
            "top_students": [],
            "n_edges_into_span": 0,
            "n_student_shifts_into_span": 0,
            "top_student_transitions": [],
        }
    sc = Counter(s for _, s, _t in events_in)
    top_s = [{"state": k, "count": v} for k, v in sc.most_common(5)]
    n_edges = 0
    shift_edges: list[tuple[str, str]] = []
    for i in range(1, len(seq)):
        g1 = seq[i][0]
        m = group_to_minute.get(g1, -10**9)
        if start_minute <= m <= end_minute:
            n_edges += 1
            _g0, s0, _t0 = seq[i - 1]
            _g1, s1, _t1 = seq[i]
            if s0 != s1:
                shift_edges.append((s0, s1))
    shift_counter = Counter(f"{a} → {b}" for a, b in shift_edges)
    top_tr = [{"edge": k, "count": v} for k, v in shift_counter.most_common(5)]
    return {
        "n_events": n_events,
        "unique_students": len(sc),
        "top_students": top_s,
        "n_edges_into_span": n_edges,
        "n_student_shifts_into_span": len(shift_edges),
        "top_student_transitions": top_tr,
    }


def split_teacher_edge(line: str) -> tuple[str, str] | None:
    for sep in (" → ", " -> "):
        if sep in line:
            a, b = line.split(sep, 1)
            return a.strip(), b.strip()
    return None


def teacher_tokens_from_nearby(nearby: list[str]) -> tuple[int, str]:
    """Distinct teacher categories appearing as endpoints of nearby edges."""
    toks: list[str] = []
    for line in nearby:
        p = split_teacher_edge(line)
        if p:
            toks.extend(p)
    uniq = sorted(set(toks))
    return len(uniq), ";".join(uniq)


def fmt_top_nodes(top: list[dict], limit: int = 5) -> str:
    if not top:
        return ""
    return "；".join(f"{x.get('state', '')}×{x.get('count', 0)}" for x in top[:limit])


def fmt_top_edges(top: list[dict], limit: int = 5) -> str:
    if not top:
        return ""
    return "；".join(f"{x.get('edge', '')}×{x.get('count', 0)}" for x in top[:limit])


def inclusive_duration_minutes(start_minute: int, end_minute: int) -> int:
    return max(end_minute - start_minute + 1, 1)


def per_minute(count: int, start_minute: int, end_minute: int) -> float:
    d = inclusive_duration_minutes(start_minute, end_minute)
    return round(count / d, 6) if d else 0.0


def span_row_base(
    task_id: str,
    lesson_title: str,
    record_type: str,
    record_index: int,
    start_minute: int,
    end_minute: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    dur = inclusive_duration_minutes(start_minute, end_minute)
    row: dict[str, Any] = {
        "task_id": task_id,
        "lesson_title": lesson_title,
        "record_type": record_type,
        "record_index": record_index,
        "start_minute": start_minute,
        "end_minute": end_minute,
        "duration_minutes_inclusive": dur,
    }
    row.update(extra)
    return row


def trend_note_b_segment(u_a: int, u_b: int, u_c: int, kind: str) -> str:
    label = "教师" if kind == "teacher" else "学生"
    if u_b > u_a and u_b > u_c:
        return f"B段{label}策略种数高于A与C（中段更分散）"
    if u_b < u_a and u_b < u_c:
        return f"B段{label}策略种数低于A与C（中段更集中）"
    return f"B段{label}策略种数未同时高于或低于两侧"


def collect_rows_for_task(
    task_id: str,
    diag: dict[str, Any],
    seq: list[tuple[int, str, str]] | None,
    group_to_minute: dict[int, int] | None,
) -> list[dict[str, Any]]:
    lesson_title = TASK_DISPLAY_LABELS.get(task_id, "")
    rows: list[dict[str, Any]] = []

    # --- Sudden jumps ---
    for ji, c in enumerate(diag.get("critical_points") or []):
        m = int(c.get("minute", -1))
        direction = c.get("direction", "")
        nearby = list(c.get("nearby_transitions") or [])
        n_edges = len(nearby)
        n_distinct, modes_list = teacher_tokens_from_nearby(nearby)
        w0, w1 = max(m - 1, 1), m + 1
        stu_summary = ""
        stu_unique = stu_shifts = stu_pm = stu_top_m = stu_top_e = stu_events = ""
        if seq and group_to_minute and m >= 0:
            st = analyze_minute_span_student_only(seq, group_to_minute, w0, w1)
            stu_events = st["n_events"]
            stu_unique = st["unique_students"]
            stu_shifts = st["n_student_shifts_into_span"]
            stu_pm = per_minute(int(stu_shifts), w0, w1)
            stu_top_m = fmt_top_nodes(st.get("top_students") or [])
            stu_top_e = fmt_top_edges(st.get("top_student_transitions") or [])
            stu_summary = (
                f"事件{stu_events}；学生策略{stu_unique}种；学侧变更{stu_shifts}次；"
                f"约{stu_pm}/分钟；Top {stu_top_m or '（无）'}"
            )
        rows.append(
            {
                "task_id": task_id,
                "lesson_title": lesson_title,
                "record_type": "sudden_jump",
                "record_index": ji,
                "start_minute": w0 if m >= 0 else "",
                "end_minute": w1 if m >= 0 else "",
                "duration_minutes_inclusive": (w1 - w0 + 1) if m >= 0 else "",
                "jump_center_minute": m,
                "jump_direction": direction,
                "jumps_source": c.get("jumps_source", ""),
                "n_nearby_teacher_shift_edges": n_edges,
                "n_distinct_teacher_categories_nearby": n_distinct,
                "teacher_categories_nearby_list": modes_list,
                "nearby_teacher_shift_edges_text": " | ".join(nearby),
                "teacher_change_pattern_summary": "邻域±1分钟内记录的师侧标签变更边（见 nearby 列）",
                "unique_teacher_strategies_in_span": "",
                "n_teacher_shifts_in_span": "",
                "teacher_shifts_per_minute": "",
                "top_teachers_marginal": "",
                "top_teacher_transition_edges": "",
                "n_events_in_span": stu_events,
                "unique_student_strategies_in_span": stu_unique,
                "n_student_shifts_in_span": stu_shifts,
                "student_shifts_per_minute": stu_pm,
                "top_students_marginal": stu_top_m,
                "top_student_transition_edges": stu_top_e,
                "student_engagement_summary": stu_summary,
                "phase_block_index": "",
                "phase_segment": "",
                "phase_slide_direction": "",
                "phase_critical_minutes": "",
                "phase_teacher_unique_A": "",
                "phase_teacher_unique_B": "",
                "phase_teacher_unique_C": "",
                "phase_student_unique_A": "",
                "phase_student_unique_B": "",
                "phase_student_unique_C": "",
                "phase_trend_note_teacher": "",
                "phase_trend_note_student": "",
            }
        )

    def emit_span(
        record_type: str,
        record_index: int,
        m0: int,
        m1: int,
        seq_: list[tuple[int, str, str]] | None,
        gtm: dict[int, int] | None,
    ) -> None:
        te = se = None
        if seq_ and gtm:
            te = analyze_minute_span_teacher_only(seq_, gtm, m0, m1)
            se = analyze_minute_span_student_only(seq_, gtm, m0, m1)
        tpm = per_minute(int(te["n_teacher_shifts_into_span"]), m0, m1) if te else ""
        spm = per_minute(int(se["n_student_shifts_into_span"]), m0, m1) if se else ""
        stu_summary = ""
        if se:
            stu_summary = (
                f"事件{se['n_events']}；学生策略{se['unique_students']}种；学侧变更{se['n_student_shifts_into_span']}次；"
                f"约{spm}/分钟；Top {fmt_top_nodes(se.get('top_students') or []) or '（无）'}"
            )
        rows.append(
            span_row_base(
                task_id,
                lesson_title,
                record_type,
                record_index,
                m0,
                m1,
                {
                    "jump_center_minute": "",
                    "jump_direction": "",
                    "jumps_source": "",
                    "n_nearby_teacher_shift_edges": "",
                    "n_distinct_teacher_categories_nearby": "",
                    "teacher_categories_nearby_list": "",
                    "nearby_teacher_shift_edges_text": "",
                    "teacher_change_pattern_summary": "",
                    "unique_teacher_strategies_in_span": te["unique_teachers"] if te else "",
                    "n_teacher_shifts_in_span": te["n_teacher_shifts_into_span"] if te else "",
                    "teacher_shifts_per_minute": tpm,
                    "top_teachers_marginal": fmt_top_nodes(te.get("top_teachers") or []) if te else "",
                    "top_teacher_transition_edges": fmt_top_edges(
                        te.get("top_teacher_transitions") or []
                    )
                    if te
                    else "",
                    "n_events_in_span": se["n_events"] if se else "",
                    "unique_student_strategies_in_span": se["unique_students"] if se else "",
                    "n_student_shifts_in_span": se["n_student_shifts_into_span"] if se else "",
                    "student_shifts_per_minute": spm,
                    "top_students_marginal": fmt_top_nodes(se.get("top_students") or []) if se else "",
                    "top_student_transition_edges": fmt_top_edges(
                        se.get("top_student_transitions") or []
                    )
                    if se
                    else "",
                    "student_engagement_summary": stu_summary,
                    "phase_block_index": "",
                    "phase_segment": "",
                    "phase_slide_direction": "",
                    "phase_critical_minutes": "",
                    "phase_teacher_unique_A": "",
                    "phase_teacher_unique_B": "",
                    "phase_teacher_unique_C": "",
                    "phase_student_unique_A": "",
                    "phase_student_unique_B": "",
                    "phase_student_unique_C": "",
                    "phase_trend_note_teacher": "",
                    "phase_trend_note_student": "",
                },
            )
        )

    bridge = diag.get("sliding_ssg_bridge") or {}

    for pi, item in enumerate(bridge.get("plateaus") or []):
        sp = (item.get("span") or {})
        try:
            m0, m1 = int(sp["start_minute"]), int(sp["end_minute"])
        except (KeyError, TypeError, ValueError):
            continue
        emit_span("plateau", pi, m0, m1, seq, group_to_minute)

    for si, item in enumerate(bridge.get("strong") or []):
        sp = (item.get("span") or {})
        try:
            m0, m1 = int(sp["start_minute"]), int(sp["end_minute"])
        except (KeyError, TypeError, ValueError):
            continue
        emit_span("strong_attractor_span", si, m0, m1, seq, group_to_minute)

    for wi, item in enumerate(bridge.get("weak") or []):
        sp = (item.get("span") or {})
        try:
            m0, m1 = int(sp["start_minute"]), int(sp["end_minute"])
        except (KeyError, TypeError, ValueError):
            continue
        emit_span("weak_attractor_span", wi, m0, m1, seq, group_to_minute)

    for pbi, ph in enumerate(bridge.get("phases") or []):
        phase_meta = ph.get("phase") or {}
        d = phase_meta.get("direction", "")
        cm = phase_meta.get("critical_minutes") or []
        cm_str = ";".join(str(x) for x in cm)

        for seg_label, span_key, ssg_key in (
            ("A", "a_span", "ssg_a"),
            ("B", "b_span", "ssg_b"),
            ("C", "c_span", "ssg_c"),
        ):
            span = phase_meta.get(span_key) or {}
            try:
                m0, m1 = int(span["start_minute"]), int(span["end_minute"])
            except (KeyError, TypeError, ValueError):
                continue
            te = ph.get(ssg_key) or {}
            se = (
                analyze_minute_span_student_only(seq, group_to_minute, m0, m1)
                if seq and group_to_minute
                else None
            )
            tpm = per_minute(int(te.get("n_teacher_shifts_into_span", 0)), m0, m1)
            spm = per_minute(int(se["n_student_shifts_into_span"]), m0, m1) if se else ""
            stu_summary = ""
            if se:
                stu_summary = (
                    f"事件{se['n_events']}；学生策略{se['unique_students']}种；学侧变更{se['n_student_shifts_into_span']}次；"
                    f"约{spm}/分钟；Top {fmt_top_nodes(se.get('top_students') or []) or '（无）'}"
                )
            rows.append(
                span_row_base(
                    task_id,
                    lesson_title,
                    f"phase_segment_{seg_label}",
                    pbi,
                    m0,
                    m1,
                    {
                        "jump_center_minute": "",
                        "jump_direction": "",
                        "jumps_source": "",
                        "n_nearby_teacher_shift_edges": "",
                        "n_distinct_teacher_categories_nearby": "",
                        "teacher_categories_nearby_list": "",
                        "nearby_teacher_shift_edges_text": "",
                        "teacher_change_pattern_summary": "",
                        "unique_teacher_strategies_in_span": te.get("unique_teachers", ""),
                        "n_teacher_shifts_in_span": te.get("n_teacher_shifts_into_span", ""),
                        "teacher_shifts_per_minute": tpm,
                        "top_teachers_marginal": fmt_top_nodes(te.get("top_teachers") or []),
                        "top_teacher_transition_edges": fmt_top_edges(
                            te.get("top_teacher_transitions") or []
                        ),
                        "n_events_in_span": se["n_events"] if se else "",
                        "unique_student_strategies_in_span": se["unique_students"] if se else "",
                        "n_student_shifts_in_span": se["n_student_shifts_into_span"] if se else "",
                        "student_shifts_per_minute": spm,
                        "top_students_marginal": fmt_top_nodes(se.get("top_students") or []) if se else "",
                        "top_student_transition_edges": fmt_top_edges(
                            se.get("top_student_transitions") or []
                        )
                        if se
                        else "",
                        "student_engagement_summary": stu_summary,
                        "phase_block_index": pbi,
                        "phase_segment": seg_label,
                        "phase_slide_direction": d,
                        "phase_critical_minutes": cm_str,
                        "phase_teacher_unique_A": "",
                        "phase_teacher_unique_B": "",
                        "phase_teacher_unique_C": "",
                        "phase_student_unique_A": "",
                        "phase_student_unique_B": "",
                        "phase_student_unique_C": "",
                        "phase_trend_note_teacher": "",
                        "phase_trend_note_student": "",
                    },
                )
            )

        sa, sb, sc = ph.get("ssg_a") or {}, ph.get("ssg_b") or {}, ph.get("ssg_c") or {}
        ua, ub, uc = int(sa.get("unique_teachers", 0)), int(sb.get("unique_teachers", 0)), int(
            sc.get("unique_teachers", 0)
        )
        ssa = (
            analyze_minute_span_student_only(
                seq, group_to_minute, int(phase_meta["a_span"]["start_minute"]), int(phase_meta["a_span"]["end_minute"])
            )
            if seq and group_to_minute and isinstance(phase_meta.get("a_span"), dict)
            else None
        )
        ssb = (
            analyze_minute_span_student_only(
                seq, group_to_minute, int(phase_meta["b_span"]["start_minute"]), int(phase_meta["b_span"]["end_minute"])
            )
            if seq and group_to_minute and isinstance(phase_meta.get("b_span"), dict)
            else None
        )
        ssc_ = (
            analyze_minute_span_student_only(
                seq, group_to_minute, int(phase_meta["c_span"]["start_minute"]), int(phase_meta["c_span"]["end_minute"])
            )
            if seq and group_to_minute and isinstance(phase_meta.get("c_span"), dict)
            else None
        )
        su_a = ssa["unique_students"] if ssa else ""
        su_b = ssb["unique_students"] if ssb else ""
        su_c = ssc_["unique_students"] if ssc_ else ""

        rows.append(
            {
                "task_id": task_id,
                "lesson_title": lesson_title,
                "record_type": "phase_transition_summary",
                "record_index": pbi,
                "start_minute": "",
                "end_minute": "",
                "duration_minutes_inclusive": "",
                "jump_center_minute": "",
                "jump_direction": "",
                "jumps_source": "",
                "n_nearby_teacher_shift_edges": "",
                "n_distinct_teacher_categories_nearby": "",
                "teacher_categories_nearby_list": "",
                "nearby_teacher_shift_edges_text": "",
                "teacher_change_pattern_summary": f"三段教师策略种数 A={ua} B={ub} C={uc}；滑动轨方向={d}",
                "unique_teacher_strategies_in_span": "",
                "n_teacher_shifts_in_span": "",
                "teacher_shifts_per_minute": "",
                "top_teachers_marginal": "",
                "top_teacher_transition_edges": "",
                "n_events_in_span": "",
                "unique_student_strategies_in_span": "",
                "n_student_shifts_in_span": "",
                "student_shifts_per_minute": "",
                "top_students_marginal": "",
                "top_student_transition_edges": "",
                "student_engagement_summary": f"三段学生策略种数 A={su_a} B={su_b} C={su_c}",
                "phase_block_index": pbi,
                "phase_segment": "ABC_summary",
                "phase_slide_direction": d,
                "phase_critical_minutes": cm_str,
                "phase_teacher_unique_A": ua,
                "phase_teacher_unique_B": ub,
                "phase_teacher_unique_C": uc,
                "phase_student_unique_A": su_a,
                "phase_student_unique_B": su_b,
                "phase_student_unique_C": su_c,
                "phase_trend_note_teacher": trend_note_b_segment(ua, ub, uc, "teacher"),
                "phase_trend_note_student": (
                    trend_note_b_segment(
                        ssa["unique_students"], ssb["unique_students"], ssc_["unique_students"], "student"
                    )
                    if ssa is not None and ssb is not None and ssc_ is not None
                    else ""
                ),
            }
        )

    return rows


FIELDNAMES = [
    "task_id",
    "lesson_title",
    "record_type",
    "record_index",
    "start_minute",
    "end_minute",
    "duration_minutes_inclusive",
    "jump_center_minute",
    "jump_direction",
    "jumps_source",
    "n_nearby_teacher_shift_edges",
    "n_distinct_teacher_categories_nearby",
    "teacher_categories_nearby_list",
    "nearby_teacher_shift_edges_text",
    "teacher_change_pattern_summary",
    "unique_teacher_strategies_in_span",
    "n_teacher_shifts_in_span",
    "teacher_shifts_per_minute",
    "top_teachers_marginal",
    "top_teacher_transition_edges",
    "n_events_in_span",
    "unique_student_strategies_in_span",
    "n_student_shifts_in_span",
    "student_shifts_per_minute",
    "top_students_marginal",
    "top_student_transition_edges",
    "student_engagement_summary",
    "phase_block_index",
    "phase_segment",
    "phase_slide_direction",
    "phase_critical_minutes",
    "phase_teacher_unique_A",
    "phase_teacher_unique_B",
    "phase_teacher_unique_C",
    "phase_student_unique_A",
    "phase_student_unique_B",
    "phase_student_unique_C",
    "phase_trend_note_teacher",
    "phase_trend_note_student",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="SSG lesson span / jump detail CSV export.")
    p.add_argument("--tasks-dir", type=Path, default=root / "tasks")
    p.add_argument(
        "--out",
        type=Path,
        default=root / "tasks" / "ssg_corpus_summary" / "ssg_lesson_span_detail.csv",
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

    all_rows: list[dict[str, Any]] = []
    for tid in ids:
        jpath = tasks_dir / tid / "ssg" / "ssg_diagnosis.json"
        try:
            diag = load_json(jpath)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skip {tid}: {e}")
            continue
        tdir = tasks_dir / tid
        built = build_seq_and_minute_map(tdir)
        seq, gtm = (built if built else (None, None))
        if not built:
            print(f"Warning: {tid} missing class/group JSON; student span stats empty.")
        all_rows.extend(collect_rows_for_task(tid, diag, seq, gtm))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})

    print(f"Wrote {args.out} ({len(all_rows)} rows, {len(ids)} tasks)")


if __name__ == "__main__":
    main()
