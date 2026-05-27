"""
Export per-task sliding dynamics to CSV (UTF-8 BOM for Excel).

All time ranges are parsed from each task's sliding/sliding_report.md (same wording
as j_sliding_report.build_markdown), so CSV rows match the human-readable report.

Minute-range columns use Excel text formulas (e.g. ="6-8") to avoid date auto-format.

Logs in English.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

from g_cluster_dynamics import TASK_DISPLAY_LABELS

# Excel: force text for range-like cells
_FIELDS_EXCEL_FORCE_TEXT: frozenset[str] = frozenset(
    {
        "ma_relative_high_minutes",
        "ma_relative_low_minutes",
        "plateau_minute_ranges",
        "sudden_jump_minutes",
        "cliff_down_minutes",
        "narrow_bw_minute_ranges",
        "wide_bw_minute_ranges",
        "strong_attractor_minute_ranges",
        "weak_attractor_minute_ranges",
        "phase_transition_b_minute_ranges",
    }
)


def _csv_cell_excel_text_literal(raw: Any) -> str:
    if raw is None or raw == "":
        return ""
    inner = str(raw).replace('"', '""')
    return f'="{inner}"'


def _normalize_hyphen(s: str) -> str:
    return (
        s.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def _parse_span_chunks_to_pipe_list(raw: str) -> tuple[int, str]:
    """
    MD fragments like '10-15分钟；19-29分钟' or '18-25分钟' -> count and '10-15|19-29'.
    """
    raw = raw.strip().rstrip("。").strip()
    if not raw or "未稳定检出" in raw:
        return 0, ""
    parts = re.split(r"[;；]", raw)
    norms: list[str] = []
    for p in parts:
        p = _normalize_hyphen(p.strip()).replace("分钟", "").strip()
        if not p:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
        if m:
            norms.append(f"{int(m.group(1))}-{int(m.group(2))}")
            continue
        m2 = re.fullmatch(r"(\d+)", p)
        if m2:
            n = int(m2.group(1))
            norms.append(f"{n}-{n}")
    return len(norms), "|".join(norms)


def _label_to_minute_range(label: str) -> str:
    """第33–35分钟 / 第4分钟 -> 33-35 / 4-4."""
    s = _normalize_hyphen(label.strip())
    s = s.removeprefix("第").replace("分钟", "").strip()
    s = _normalize_hyphen(s)
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    m2 = re.fullmatch(r"(\d+)", s)
    if m2:
        n = int(m2.group(1))
        return f"{n}-{n}"
    return s


def parse_sliding_report_md(text: str) -> dict[str, Any]:
    """Extract CSV fields from sliding_report.md body."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out: dict[str, Any] = {
        "report_task_id": "",
        "ma_relative_high_minutes": "",
        "ma_relative_low_minutes": "",
        "plateau_segment_count": 0,
        "plateau_minute_ranges": "",
        "sudden_jump_minutes": "",
        "cliff_down_minutes": "",
        "narrow_bw_segment_count": 0,
        "narrow_bw_minute_ranges": "",
        "wide_bw_segment_count": 0,
        "wide_bw_minute_ranges": "",
        "strong_attractor_count": 0,
        "strong_attractor_minute_ranges": "",
        "weak_attractor_count": 0,
        "weak_attractor_minute_ranges": "",
        "has_phase_transition": "否",
        "phase_transition_b_minute_ranges": "",
    }

    m_id = re.search(r"课程编号\*\*[：:]\s*`([^`]+)`", text)
    if m_id:
        out["report_task_id"] = m_id.group(1).strip()

    # Moving mean high / low (matches window_label: 第a分钟 or 第a–b分钟)
    m_hi = re.search(
        r"移动均值[的]?相对高点在\s*\*\*(.+?)\*\*\s*，\s*相对低点在\s*\*\*(.+?)\*\*\s*。",
        text,
    )
    if m_hi:
        out["ma_relative_high_minutes"] = _label_to_minute_range(m_hi.group(1))
        out["ma_relative_low_minutes"] = _label_to_minute_range(m_hi.group(2))

    # Plateau: 在 **...** 等区间
    m_plat = re.search(r"\*\*平台期识别\*\*[：:]\s*\n在 \*\*([^*]+)\*\* 等区间", text)
    if m_plat:
        c, s = _parse_span_chunks_to_pipe_list(m_plat.group(1))
        out["plateau_segment_count"] = c
        out["plateau_minute_ranges"] = s

    # Sudden jumps paragraph (section 一)
    # Greedy body: a non-greedy .+? would stop at the first blank line inside section 一.
    m_sec1 = re.search(r"## 一、总体发展趋势\s*(.+)(?=\n## 二、)", text, re.DOTALL)
    if m_sec1:
        block1 = m_sec1.group(1)
        m_jump = re.search(r"\*\*突然跃迁与异常点\*\*[：:]\s*\n(.+)", block1, re.DOTALL)
        if m_jump:
            chunk = m_jump.group(1)
            para: list[str] = []
            for ln in chunk.split("\n"):
                if not ln.strip():
                    break
                if ln.strip().startswith("##"):
                    break
                para.append(ln)
            jline = re.sub(r"\s+", " ", " ".join(para)).strip()
            if "未检出显著的梯度尖峰" in jline:
                pass
            else:
                up_mins: list[str] = []
                if "上升" in jline:
                    pre_up = jline.split("上升", 1)[0]
                    if "其中在" in pre_up:
                        pre_up = pre_up.split("其中在", 1)[1]
                    up_mins = re.findall(r"第(\d+)分钟", pre_up)
                down_mins: list[str] = []
                if "断崖式下降" in jline:
                    pre_dn = jline.split("断崖式下降", 1)[0]
                    if "而在" in pre_dn:
                        pre_dn = pre_dn.split("而在", 1)[1]
                    down_mins = re.findall(r"第(\d+)分钟", pre_dn)
                # Report splits explicitly: 上升 vs 断崖式下降 — no duplicate across columns.
                out["sudden_jump_minutes"] = ",".join(
                    str(x) for x in sorted({int(x) for x in up_mins})
                )
                out["cliff_down_minutes"] = ",".join(
                    str(x) for x in sorted({int(x) for x in down_mins})
                )

    # Narrow / wide bandwidth bullets
    m_narrow = re.search(
        r"- \*\*极窄带宽（高稳定性）\*\*[：:](.+?)。\s*带宽极窄",
        text,
        re.DOTALL,
    )
    if m_narrow:
        c, s = _parse_span_chunks_to_pipe_list(m_narrow.group(1))
        out["narrow_bw_segment_count"] = c
        out["narrow_bw_minute_ranges"] = s

    m_wide = re.search(
        r"- \*\*极宽带宽（高变异性）\*\*[：:](.+?)。\s*带宽拉宽",
        text,
        re.DOTALL,
    )
    if m_wide:
        c, s = _parse_span_chunks_to_pipe_list(m_wide.group(1))
        out["wide_bw_segment_count"] = c
        out["wide_bw_minute_ranges"] = s

    m_strong = re.search(r"- \*\*强吸引子时段\*\*[：:]([^\n]+)", text)
    if m_strong:
        raw_s = m_strong.group(1).strip().rstrip("。")
        c, s = _parse_span_chunks_to_pipe_list(raw_s)
        out["strong_attractor_count"] = c
        out["strong_attractor_minute_ranges"] = s

    m_weak = re.search(r"- \*\*弱吸引子时段\*\*[：:]([^\n]+)", text)
    if m_weak:
        raw_w = m_weak.group(1).strip().rstrip("。")
        c, s = _parse_span_chunks_to_pipe_list(raw_w)
        out["weak_attractor_count"] = c
        out["weak_attractor_minute_ranges"] = s

    m_s4 = re.search(r"## 四、临界点与相变定位\s*\n(.+)(?=\n---)", text, re.DOTALL)
    if m_s4:
        s4 = m_s4.group(1)
        if "本课数据上 **未检出**" in s4:
            out["has_phase_transition"] = "否"
            out["phase_transition_b_minute_ranges"] = ""
        else:
            pairs = re.findall(
                r"鼓包，段 B）\*\*[：:]\s*约 \*\*(\d+)\s*[-\u2013–]\s*(\d+)\s*分钟\*\*",
                s4,
            )
            if pairs:
                out["has_phase_transition"] = "是"
                out["phase_transition_b_minute_ranges"] = "|".join(
                    f"{int(a)}-{int(b)}" for a, b in pairs
                )
            else:
                out["has_phase_transition"] = "否"
                out["phase_transition_b_minute_ranges"] = ""

    return out


def discover_task_ids(tasks_dir: Path) -> list[str]:
    out: list[str] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "sliding" / "sliding_report.md").is_file():
            out.append(d.name)
    return out


def build_row(task_dir: Path, md_text: str) -> dict[str, Any]:
    tid = task_dir.name
    parsed = parse_sliding_report_md(md_text)
    report_id = parsed.pop("report_task_id", "")
    if report_id and report_id != tid:
        print(f"Warn: report task id {report_id!r} != folder {tid!r}")

    row: dict[str, Any] = {
        "task_id": tid,
        "lesson_title": TASK_DISPLAY_LABELS.get(tid, ""),
        **parsed,
    }
    return row


def csv_fieldnames() -> list[str]:
    return [
        "task_id",
        "lesson_title",
        "ma_relative_high_minutes",
        "ma_relative_low_minutes",
        "plateau_segment_count",
        "plateau_minute_ranges",
        "sudden_jump_minutes",
        "cliff_down_minutes",
        "narrow_bw_segment_count",
        "narrow_bw_minute_ranges",
        "wide_bw_segment_count",
        "wide_bw_minute_ranges",
        "strong_attractor_count",
        "strong_attractor_minute_ranges",
        "weak_attractor_count",
        "weak_attractor_minute_ranges",
        "has_phase_transition",
        "phase_transition_b_minute_ranges",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sliding metrics from sliding_report.md to CSV.",
    )
    parser.add_argument(
        "task_ids",
        nargs="*",
        help="Task IDs (default: all tasks with sliding/sliding_report.md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: tasks/sliding_aggregate_stats.csv)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tasks_dir = repo_root / "tasks"
    if not tasks_dir.is_dir():
        print("tasks/ not found")
        sys.exit(1)

    if args.task_ids:
        ids = list(args.task_ids)
    else:
        ids = discover_task_ids(tasks_dir)

    if not ids:
        print("No tasks with sliding/sliding_report.md")
        sys.exit(1)

    out_path = args.output or (tasks_dir / "sliding_aggregate_stats.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for tid in ids:
        task_dir = tasks_dir / tid
        md_path = task_dir / "sliding" / "sliding_report.md"
        if not md_path.is_file():
            print(f"Skip {tid}: missing {md_path}")
            continue
        try:
            md_text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skip {tid}: failed to read {md_path} ({e})")
            continue
        rows.append(build_row(task_dir, md_text))

    fields = csv_fieldnames()
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            line: dict[str, Any] = {}
            for k in fields:
                v = r.get(k, "")
                if k in _FIELDS_EXCEL_FORCE_TEXT:
                    line[k] = _csv_cell_excel_text_literal(v)
                else:
                    line[k] = v
            w.writerow(line)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
