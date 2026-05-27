import json
import math
import statistics
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


TEACHER_RESULT_ALIASES = {
    "反思": "反思对话或活动",
    "12维持秩序或与课堂无关的内容": "维持秩序或与课堂无关的内容",
    "11 (讲授)": "讲授",
    "9 (Guide direction - 引导方向)": "引导方向",
    "10 Express/Invite ideas - 表达或邀请想法": "表达或邀请想法",
    "7 Connect - 联系": "联系",
    "2 Build on ideas - 邀请补充想法": "补充想法",
    "1 IB - 教师邀请学生补充发展想法": "邀请学生补充想法",
    "5 Reasoning - 进行明确的推理论证": "进行明确的推理论证",
    "4 IRE - 教师邀请学生推理论证": "邀请学生推理论证",
    "3 Challenge - 质疑": "质疑",
    "6 Coordinate - 想法上的协调和同意": "想法上的协调和同意",
    "8 Reflect - 反思": "反思对话或活动",
    "8 Reflect - 反思对话或活动": "反思对话或活动",
}

FIXED_TEACHER_CATEGORIES = [
    "讲授",
    "引导方向",
    "表达或邀请想法",
    "联系",
    "邀请学生补充想法",
    "邀请学生推理论证",
    "质疑",
    "进行明确的推理论证",
    "想法上的协调和同意",
    "补充想法",
    "反思对话或活动",
    "维持秩序或与课堂无关的内容",
]

FIXED_STUDENT_CATEGORIES = [
    "接受",
    "记忆",
    "应用",
    "提问",
    "阐述",
    "创造",
    "支持",
    "反对",
    "讨论",
]


def resolve_task_id() -> str:
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        return sys.argv[1].strip()
    tasks_dir = Path("tasks")
    if not tasks_dir.exists():
        print("Tasks directory not found.")
        sys.exit(1)
    task_dirs = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
    if not task_dirs:
        print("No tasks found.")
        sys.exit(1)
    task_id = task_dirs[-1]
    print(f"No task_id provided, using latest: {task_id}")
    return task_id


def normalize_teacher_result(raw: str) -> str:
    return TEACHER_RESULT_ALIASES.get(raw, raw)


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"Missing file: {path}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def load_group_bounds(group_full: dict) -> tuple[dict[int, tuple[float, float]], float]:
    bounds = defaultdict(lambda: [float("inf"), float("-inf")])
    for seg in group_full.get("segments", []):
        gid = int(seg["group"])
        bounds[gid][0] = min(bounds[gid][0], float(seg["start"]))
        bounds[gid][1] = max(bounds[gid][1], float(seg["end"]))
    clean = {}
    max_end = 0.0
    for gid, (t0, t1) in bounds.items():
        if math.isfinite(t0) and math.isfinite(t1):
            clean[gid] = (t0, t1)
            max_end = max(max_end, t1)
    return clean, max_end


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    rank = (len(s) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    w = rank - lo
    return s[lo] * (1 - w) + s[hi] * w


def _normalize_jump_items(raw: object) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            m, d = item.get("minute"), item.get("direction")
            if isinstance(m, (int, float)) and d in ("up", "down"):
                out.append({"minute": int(m), "direction": d})
    return out


def load_sliding_insights(task_dir: Path, windows: list[dict]) -> tuple[dict, str]:
    """
    优先读取 `sliding/sliding_insights.json`（j_sliding_report 写的单一侧车）；
    否则尝试 legacy `sliding_jumps.json` 并与现场补全 plateaus 等；
    再否则整份由 j_sliding_report 同算法现场计算。
    """
    from j_sliding_report import build_sliding_insights_document

    p_ins = task_dir / "sliding" / "sliding_insights.json"
    if p_ins.is_file():
        try:
            doc = json.loads(p_ins.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
        if (
            isinstance(doc, dict)
            and isinstance(doc.get("jumps"), list)
            and isinstance(doc.get("plateaus"), list)
        ):
            doc = dict(doc)
            doc["jumps"] = _normalize_jump_items(doc["jumps"])
            return doc, "sliding_insights.json"

    p_old = task_dir / "sliding" / "sliding_jumps.json"
    if p_old.is_file():
        try:
            old = json.loads(p_old.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
        merged = build_sliding_insights_document(windows)
        jn = _normalize_jump_items(old.get("jumps"))
        if jn:
            merged["jumps"] = jn
        return merged, "sliding_jumps.json_legacy"

    return build_sliding_insights_document(windows), "computed_fallback"


def analyze_minute_span_teacher_only(
    seq: list[tuple[int, str, str]],
    group_to_minute: dict[int, int],
    start_minute: int,
    end_minute: int,
) -> dict:
    """
    仅聚合 **教师言语策略**：区间内事件上的教师标签边际分布、
    落入区间边中的 **教师策略变更** 次数及 Top 师侧转移（t0→t1）。
    不展开学生认知标签，仅看教师侧聚合。
    """
    events_in = [
        (gid, s, t)
        for gid, s, t in seq
        if start_minute <= group_to_minute.get(gid, -10**9) <= end_minute
    ]
    n_events = len(events_in)
    if n_events == 0:
        return {
            "n_events": 0,
            "unique_teachers": 0,
            "top_teachers": [],
            "n_edges_into_span": 0,
            "n_teacher_shifts_into_span": 0,
            "top_teacher_transitions": [],
        }
    tc = Counter(t for _, _s, t in events_in)
    top_t = [{"state": k, "count": v} for k, v in tc.most_common(5)]
    n_edges = 0
    shift_edges: list[tuple[str, str]] = []
    for i in range(1, len(seq)):
        g1 = seq[i][0]
        m = group_to_minute.get(g1, -10**9)
        if start_minute <= m <= end_minute:
            n_edges += 1
            _g0, _s0, t0 = seq[i - 1]
            _g1, _s1, t1 = seq[i]
            if t0 != t1:
                shift_edges.append((t0, t1))
    shift_counter = Counter(f"{a} → {b}" for a, b in shift_edges)
    top_tr = [{"edge": k, "count": v} for k, v in shift_counter.most_common(5)]
    return {
        "n_events": n_events,
        "unique_teachers": len(tc),
        "top_teachers": top_t,
        "n_edges_into_span": n_edges,
        "n_teacher_shifts_into_span": len(shift_edges),
        "top_teacher_transitions": top_tr,
    }


def build_sliding_ssg_bridge(
    seq: list[tuple[int, str, str]],
    group_to_minute: dict[int, int],
    insights: dict,
) -> dict:
    """为平台期 / 滑动轨吸引子时段 / 相变三段 附加 **教师策略** 粗统计。"""

    def span_rows(key: str) -> list[dict]:
        rows: list[dict] = []
        for sp in insights.get(key, []) or []:
            if not isinstance(sp, dict):
                continue
            try:
                m0 = int(sp["start_minute"])
                m1 = int(sp["end_minute"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "span": {"start_minute": m0, "end_minute": m1},
                    "ssg": analyze_minute_span_teacher_only(seq, group_to_minute, m0, m1),
                }
            )
        return rows

    phase_rows: list[dict] = []
    for ph in insights.get("phase_transitions", []) or []:
        if not isinstance(ph, dict):
            continue
        try:
            a = ph["a_span"]
            b = ph["b_span"]
            c = ph["c_span"]
            phase_rows.append(
                {
                    "phase": ph,
                    "ssg_a": analyze_minute_span_teacher_only(
                        seq, group_to_minute, int(a["start_minute"]), int(a["end_minute"])
                    ),
                    "ssg_b": analyze_minute_span_teacher_only(
                        seq, group_to_minute, int(b["start_minute"]), int(b["end_minute"])
                    ),
                    "ssg_c": analyze_minute_span_teacher_only(
                        seq, group_to_minute, int(c["start_minute"]), int(c["end_minute"])
                    ),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "plateaus": span_rows("plateaus"),
        "strong": span_rows("strong_attractor_spans"),
        "weak": span_rows("weak_attractor_spans"),
        "phases": phase_rows,
    }


def _fmt_ssg_top_nodes(top: list[dict], limit: int = 3) -> str:
    if not top:
        return "（无）"
    parts: list[str] = []
    for x in top[:limit]:
        parts.append(f"`{x.get('state', '')}`×{x.get('count', 0)}")
    return "；".join(parts)


def _fmt_teacher_trans_top(top: list[dict], limit: int = 3) -> str:
    if not top:
        return "（无）"
    parts: list[str] = []
    for x in top[:limit]:
        parts.append(f"`{x.get('edge', '')}`×{x.get('count', 0)}")
    return "；".join(parts)


def coupling_direction(seq: list[tuple[int, str, str]]) -> dict:
    if len(seq) < 3:
        return {
            "synchronous": 0,
            "teacher_leads": 0,
            "student_leads": 0,
            "ambiguous": 0,
            "dominant_mode": "insufficient_data",
        }

    teacher_changed = [False]
    student_changed = [False]
    for i in range(1, len(seq)):
        _, s_prev, t_prev = seq[i - 1]
        _, s_cur, t_cur = seq[i]
        student_changed.append(s_prev != s_cur)
        teacher_changed.append(t_prev != t_cur)

    sync = 0
    t_lead = 0
    s_lead = 0
    amb = 0
    for i in range(1, len(seq) - 1):
        tc = teacher_changed[i]
        sc = student_changed[i]
        tc_next = teacher_changed[i + 1]
        sc_next = student_changed[i + 1]
        if tc and sc:
            sync += 1
        elif tc and not sc and sc_next:
            t_lead += 1
        elif sc and not tc and tc_next:
            s_lead += 1
        elif tc_next and sc_next:
            sync += 1
        else:
            amb += 1

    stats = {
        "synchronous": sync,
        "teacher_leads": t_lead,
        "student_leads": s_lead,
        "ambiguous": amb,
    }
    dominant = max(stats, key=stats.get)
    stats["dominant_mode"] = dominant
    return stats


def entropy_from_probs(probs: list[float]) -> float:
    """Shannon entropy with natural log: Σ(P * ln(1/P))."""
    h = 0.0
    for p in probs:
        if p > 0:
            h += p * math.log(1.0 / p)
    return h


def normalized_entropy(entropy_value: float, bucket_count: int) -> float:
    if bucket_count <= 1:
        return 0.0
    return entropy_value / math.log(bucket_count)


# 固定 9×12 状态空间，用于转换熵归一化上界（与文献/GridWare 中「可能下一状态」规模一致）
_ENTROPY_TRANSITION_NORM_BUCKETS = 108


def transitional_entropy_conditional(transition_counts: Counter) -> tuple[float, Counter]:
    """文献口径「转换熵」：条件熵 H(Y|X)，自然对数。

    对每个起点 X：P(Y|X)=N_{X→Y}/N_out(X)；H(Y|X=X)=Σ_y P(y|x)ln(1/P(y|x))。
    再按「从 X 出发的转移」占比加权：H(Y|X)=Σ_x (N_out(x)/N_total)·H(Y|X=x)。
    """
    outgoing: Counter = Counter()
    for (frm, _to), cnt in transition_counts.items():
        outgoing[frm] += cnt
    total = float(sum(transition_counts.values()))
    if total <= 0:
        return 0.0, outgoing
    h = 0.0
    for frm, n_out in outgoing.items():
        if n_out <= 0:
            continue
        p_depart = n_out / total
        cond_probs: list[float] = []
        for (f, _to), cnt in transition_counts.items():
            if f != frm:
                continue
            cond_probs.append(cnt / n_out)
        h += p_depart * entropy_from_probs(cond_probs)
    return h, outgoing


def classify_entropy_level(norm_value: float) -> str:
    if norm_value < 0.33:
        return "低"
    if norm_value < 0.67:
        return "中"
    return "高"


# 全库四分位分级至少需要其它课程样本数（不含当前任务）
_MIN_CORPUS_TASKS_FOR_QUARTILE_LEVELS = 4


def gather_corpus_entropy_from_tasks(
    tasks_dir: Path,
    exclude_task_id: str | None,
) -> dict[str, list[float]]:
    """扫描 tasks/*/ssg/ssg_diagnosis.json，收集各课 overall 熵（原始 ln 熵，非 norm）。"""
    keys = ("visited_entropy", "duration_entropy", "transition_entropy")
    out: dict[str, list[float]] = {k: [] for k in keys}
    if not tasks_dir.is_dir():
        return out
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.is_dir():
            continue
        if exclude_task_id and task_path.name == exclude_task_id:
            continue
        jpath = task_path / "ssg" / "ssg_diagnosis.json"
        if not jpath.is_file():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            eo = data.get("entropy", {}).get("overall", {})
            for k in keys:
                v = eo.get(k)
                if v is None:
                    continue
                out[k].append(float(v))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def entropy_quartiles_q123(values: list[float]) -> tuple[float, float, float] | None:
    """Q1、Q2（中位数）、Q3；样本过少返回 None。"""
    if len(values) < _MIN_CORPUS_TASKS_FOR_QUARTILE_LEVELS:
        return None
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return float(qs[0]), float(qs[1]), float(qs[2])


def classify_entropy_level_by_quartiles(x: float, q1: float, q2: float) -> str:
    """箱线图式三档（基于原始熵值与全库 Q1、Q2；Q3 仅写入 JSON 供对照）。

    - 低：第一四分位及以下（x ≤ Q1）
    - 中：第二四分位（Q1 < x ≤ Q2，Q2 为中位数）
    - 高：第三四分位及以上（x > Q2，含第三与第四四分位）
    """
    if x <= q1:
        return "低"
    if x <= q2:
        return "中"
    return "高"


def build_entropy_metrics(
    visits: Counter,
    duration: Counter,
    transition_counts: Counter,
    corpus_entropy_quartiles: dict[str, tuple[float, float, float]] | None = None,
    corpus_entropy_sample_sizes: dict[str, int] | None = None,
) -> dict:
    total_visits = float(sum(visits.values()))
    total_duration = float(sum(duration.values()))
    total_transitions = float(sum(transition_counts.values()))

    visited_probs = {st: (cnt / total_visits if total_visits > 0 else 0.0) for st, cnt in visits.items()}
    duration_probs = {st: (dur / total_duration if total_duration > 0 else 0.0) for st, dur in duration.items()}
    transition_probs_joint = {
        (frm, to): (cnt / total_transitions if total_transitions > 0 else 0.0)
        for (frm, to), cnt in transition_counts.items()
    }
    transition_entropy, outgoing = transitional_entropy_conditional(transition_counts)

    visited_entropy = entropy_from_probs(list(visited_probs.values()))
    duration_entropy = entropy_from_probs(list(duration_probs.values()))

    visited_norm = normalized_entropy(visited_entropy, len(visited_probs))
    duration_norm = normalized_entropy(duration_entropy, len(duration_probs))
    transition_norm = normalized_entropy(
        transition_entropy, _ENTROPY_TRANSITION_NORM_BUCKETS
    )

    def _level_for(
        metric_key: str,
        raw_entropy: float,
        norm_value: float,
    ) -> tuple[str, str, dict | None]:
        if corpus_entropy_quartiles and metric_key in corpus_entropy_quartiles:
            q1, q2, q3 = corpus_entropy_quartiles[metric_key]
            lvl = classify_entropy_level_by_quartiles(raw_entropy, q1, q2)
            n = (corpus_entropy_sample_sizes or {}).get(metric_key, 0)
            ref = {
                "q1": round(q1, 6),
                "q2": round(q2, 6),
                "q3": round(q3, 6),
                "n_corpus_tasks": int(n),
                "rule": "<=Q1低; Q1<x<=Q2中; >Q2高 (Q2=median)",
            }
            return lvl, "corpus_quartiles_raw_entropy", ref
        return classify_entropy_level(norm_value), "normalized_entropy_tertiles", None

    v_lvl, v_meth, v_ref = _level_for(
        "visited_entropy", visited_entropy, visited_norm
    )
    d_lvl, d_meth, d_ref = _level_for(
        "duration_entropy", duration_entropy, duration_norm
    )
    t_lvl, t_meth, t_ref = _level_for(
        "transition_entropy", transition_entropy, transition_norm
    )

    transition_rows: list[dict] = []
    for (frm, to), p_joint in sorted(
        transition_probs_joint.items(), key=lambda x: x[1], reverse=True
    ):
        n_out = outgoing[frm]
        cnt = transition_counts[(frm, to)]
        p_cond = (cnt / n_out) if n_out > 0 else 0.0
        transition_rows.append(
            {
                "from_state": frm,
                "to_state": to,
                "transition": f"{frm} -> {to}",
                "count": int(cnt),
                "p_transition": round(p_cond, 6),
                "p_joint": round(p_joint, 6),
                "entropy_term": round(p_cond * math.log(1.0 / p_cond), 6)
                if p_cond > 0
                else 0.0,
            }
        )

    return {
        "overall": {
            "visited_entropy": round(visited_entropy, 6),
            "duration_entropy": round(duration_entropy, 6),
            "transition_entropy": round(transition_entropy, 6),
            "visited_entropy_norm": round(visited_norm, 6),
            "duration_entropy_norm": round(duration_norm, 6),
            "transition_entropy_norm": round(transition_norm, 6),
            "visited_entropy_level": v_lvl,
            "duration_entropy_level": d_lvl,
            "transition_entropy_level": t_lvl,
            "visited_entropy_level_method": v_meth,
            "duration_entropy_level_method": d_meth,
            "transition_entropy_level_method": t_meth,
            "visited_entropy_level_quartiles_ref": v_ref,
            "duration_entropy_level_quartiles_ref": d_ref,
            "transition_entropy_level_quartiles_ref": t_ref,
            "transition_entropy_definition": (
                "H(Y|X): weighted conditional entropy over departures; "
                "P(Y|X)=count(X→Y)/outgoing(X); ln; norm÷ln(108)"
            ),
        },
        "visited": [
            {
                "state": st,
                "visits": int(visits[st]),
                "p_visit": round(p, 6),
                "entropy_term": round(p * math.log(1.0 / p), 6) if p > 0 else 0.0,
            }
            for st, p in sorted(visited_probs.items(), key=lambda x: x[1], reverse=True)
        ],
        "duration": [
            {
                "state": st,
                "duration": round(float(duration[st]), 4),
                "p_duration": round(p, 6),
                "entropy_term": round(p * math.log(1.0 / p), 6) if p > 0 else 0.0,
            }
            for st, p in sorted(duration_probs.items(), key=lambda x: x[1], reverse=True)
        ],
        "transition": transition_rows,
    }


def build_binned_teacher_student_sequences(
    seq: list[tuple[int, str, str]],
    bounds: dict[int, tuple[float, float]],
) -> tuple[list[str], list[str], float]:
    """按最小事件时长分箱，展开教师与学生两个等时间步序列。"""
    if not seq:
        return [], [], 0.0
    intervals: list[tuple[float, float, str, str]] = []
    min_dur = None
    for gid, s, t in seq:
        if gid not in bounds:
            continue
        t0, t1 = bounds[gid]
        dur = max(float(t1 - t0), 1e-6)
        min_dur = dur if min_dur is None else min(min_dur, dur)
        intervals.append((float(t0), float(t1), t, s))
    if not intervals:
        return [], [], 0.0
    bin_size = min_dur if min_dur is not None else 1.0
    teacher_bins: list[str] = []
    student_bins: list[str] = []
    for t0, t1, teacher_state, student_state in intervals:
        bins = max(1, int(round((t1 - t0) / bin_size)))
        teacher_bins.extend([teacher_state] * bins)
        student_bins.extend([student_state] * bins)
    return teacher_bins, student_bins, float(bin_size)


def _sort_tp_rows_and_group_by_from(
    rows: list[dict], from_key: str
) -> tuple[list[dict], dict[str, list[dict]]]:
    rows_sorted = sorted(
        rows,
        key=lambda x: (x["tp"], x["count"], str(sorted(x.items()))),
        reverse=True,
    )
    by_from: dict[str, list[dict]] = defaultdict(list)
    for row in rows_sorted:
        by_from[row[from_key]].append(row)
    by_from_top3: dict[str, list[dict]] = {}
    for fs, rlist in by_from.items():
        r2 = sorted(rlist, key=lambda x: (x["tp"], x["count"]), reverse=True)
        by_from_top3[fs] = r2[:3]
    return rows_sorted[:10], by_from_top3


def build_cross_role_transition_tp(
    teacher_bins: list[str], student_bins: list[str], n: int
) -> tuple[
    tuple[list[dict], dict[str, list[dict]]],
    tuple[list[dict], dict[str, list[dict]]],
]:
    """跨角色 TP，均在同一分箱时刻 t 上定义（非 t→t+1）。

    教师 A→学生 B：TP ≈ P(学生=B|教师=A) = 同箱共现箱数(教师=A 且学生=B) / 教师=A 的箱数；
    分母即文献中的 duration(教师 A)/bin_size（离散为 A 所占箱数）。

    学生 A→教师 B：TP ≈ P(教师=B|学生=A)，分母为学生=A 的箱数。
    """
    tb = teacher_bins[:n]
    sb = student_bins[:n]
    teacher_bins_per_state: Counter = Counter(tb)
    student_bins_per_state: Counter = Counter(sb)

    ts_counts: Counter = Counter()
    for i in range(n):
        ts_counts[(tb[i], sb[i])] += 1
    ts_rows: list[dict] = []
    for (t_a, s_b), cnt in ts_counts.items():
        denom = teacher_bins_per_state[t_a]
        tp = (cnt / denom) if denom > 0 else 0.0
        ts_rows.append(
            {
                "teacher_state": t_a,
                "student_state": s_b,
                "count": int(cnt),
                "bins_in_from_state": int(denom),
                "tp": round(tp, 6),
            }
        )
    ts_top, ts_by_teacher = _sort_tp_rows_and_group_by_from(ts_rows, "teacher_state")

    st_counts: Counter = Counter()
    for i in range(n):
        st_counts[(sb[i], tb[i])] += 1
    st_rows: list[dict] = []
    for (s_a, t_b), cnt in st_counts.items():
        denom = student_bins_per_state[s_a]
        tp = (cnt / denom) if denom > 0 else 0.0
        st_rows.append(
            {
                "student_state": s_a,
                "teacher_state": t_b,
                "count": int(cnt),
                "bins_in_from_state": int(denom),
                "tp": round(tp, 6),
            }
        )
    st_top, st_by_student = _sort_tp_rows_and_group_by_from(st_rows, "student_state")

    return (ts_top, ts_by_teacher), (st_top, st_by_student)


def build_cross_variable_transition_analysis(teacher_bins: list[str], student_bins: list[str]) -> dict:
    """跨角色 TP：同一箱 t 上的条件比例（教师为 A 时学生为 B；学生为 A 时教师为 B）。

    跨变量同期网格四块（lag_grid_* / reverse_lag_*）仍为 (教师_t,学生_t) 节点与共现时间路径，逻辑未改。
    """
    n = min(len(teacher_bins), len(student_bins))
    if n < 1:
        return {
            "bin_size_sec": 0.0,
            "bins": n,
            "teacher_to_student_tp_top": [],
            "teacher_to_student_tp_by_teacher": {},
            "student_to_teacher_tp_top": [],
            "student_to_teacher_tp_by_student": {},
            "lag_grid_nodes_top": [],
            "lag_grid_paths_top": [],
            "reverse_lag_grid_nodes_top": [],
            "reverse_lag_grid_paths_top": [],
        }

    lag_nodes: list[tuple[str, str]] = []
    for i in range(n):
        lag_nodes.append((teacher_bins[i], student_bins[i]))

    (teacher_to_student_top, by_teacher), (student_to_teacher_top, by_student) = (
        build_cross_role_transition_tp(teacher_bins, student_bins, n)
    )

    lag_node_counts: Counter = Counter(lag_nodes)
    lag_grid_nodes_top = []
    for (teacher_state, student_state), cnt in lag_node_counts.most_common(10):
        lag_grid_nodes_top.append(
            {
                "node": f"({teacher_state}, {student_state})",
                "teacher_state_t": teacher_state,
                "student_state_t": student_state,
                "count": int(cnt),
            }
        )

    lag_edge_counts: Counter = Counter()
    for i in range(len(lag_nodes) - 1):
        n1 = lag_nodes[i]
        n2 = lag_nodes[i + 1]
        lag_edge_counts[(n1, n2)] += 1
    lag_grid_paths_top = []
    for ((t1, s1), (t2, s2)), cnt in lag_edge_counts.most_common(10):
        lag_grid_paths_top.append(
            {
                "from_node": f"({t1}, {s1})",
                "to_node": f"({t2}, {s2})",
                "path": f"({t1}, {s1}) -> ({t2}, {s2})",
                "count": int(cnt),
            }
        )

    reverse_lag_nodes: list[tuple[str, str]] = []
    for i in range(n):
        reverse_lag_nodes.append((student_bins[i], teacher_bins[i]))
    reverse_lag_node_counts: Counter = Counter(reverse_lag_nodes)
    reverse_lag_grid_nodes_top = []
    for (student_state, teacher_state), cnt in reverse_lag_node_counts.most_common(10):
        reverse_lag_grid_nodes_top.append(
            {
                "node": f"({student_state}, {teacher_state})",
                "student_state_t": student_state,
                "teacher_state_t": teacher_state,
                "count": int(cnt),
            }
        )

    reverse_lag_edge_counts: Counter = Counter()
    for i in range(len(reverse_lag_nodes) - 1):
        n1 = reverse_lag_nodes[i]
        n2 = reverse_lag_nodes[i + 1]
        reverse_lag_edge_counts[(n1, n2)] += 1
    reverse_lag_grid_paths_top = []
    for ((s1, t1), (s2, t2)), cnt in reverse_lag_edge_counts.most_common(10):
        reverse_lag_grid_paths_top.append(
            {
                "from_node": f"({s1}, {t1})",
                "to_node": f"({s2}, {t2})",
                "path": f"({s1}, {t1}) -> ({s2}, {t2})",
                "count": int(cnt),
            }
        )

    return {
        "bins": n,
        "teacher_to_student_tp_top": teacher_to_student_top,
        "teacher_to_student_tp_by_teacher": by_teacher,
        "student_to_teacher_tp_top": student_to_teacher_top,
        "student_to_teacher_tp_by_student": by_student,
        "lag_grid_nodes_top": lag_grid_nodes_top,
        "lag_grid_paths_top": lag_grid_paths_top,
        "reverse_lag_grid_nodes_top": reverse_lag_grid_nodes_top,
        "reverse_lag_grid_paths_top": reverse_lag_grid_paths_top,
    }


def lewis_heterogeneity(cell_durations: list[float]) -> float:
    """Lewis et al. (1999) 首篇 SSG：零假设为各格观测（时长）均匀，E_j=总时长/格数。

    H_j = (1/n_j) * Σ_i (O_i - E_j)² / E_j（与文中 chi-square 型异质性一致）。
    """
    n = len(cell_durations)
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.0
    total = float(sum(cell_durations))
    e_j = total / n
    if e_j <= 1e-15:
        return 0.0
    chi_sum = sum((float(o) - e_j) ** 2 / e_j for o in cell_durations)
    return chi_sum / n


def attractor_region_ids_4neighbors(attractor_states: list[str]) -> dict[str, int]:
    """在 9×12 固定格上用四邻接划分吸引子连通片，用于判断是否「单一大吸引区」。"""
    aset = set(attractor_states)
    idx_of: dict[str, tuple[int, int]] = {}
    for st in attractor_states:
        if " | " not in st:
            continue
        s_lab, t_lab = st.split(" | ", 1)
        try:
            si = FIXED_STUDENT_CATEGORIES.index(s_lab)
            ti = FIXED_TEACHER_CATEGORIES.index(t_lab)
        except ValueError:
            continue
        idx_of[st] = (si, ti)
    visited: set[str] = set()
    regions: dict[str, int] = {}
    rid = 0
    for st in attractor_states:
        if st in visited:
            continue
        rid += 1
        stack = [st]
        while stack:
            u = stack.pop()
            if u in visited or u not in aset:
                continue
            visited.add(u)
            regions[u] = rid
            if u not in idx_of:
                continue
            si, ti = idx_of[u]
            for dsi, dti in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nsi, nti = si + dsi, ti + dti
                if not (0 <= nsi < len(FIXED_STUDENT_CATEGORIES)):
                    continue
                if not (0 <= nti < len(FIXED_TEACHER_CATEGORIES)):
                    continue
                v = f"{FIXED_STUDENT_CATEGORIES[nsi]} | {FIXED_TEACHER_CATEGORIES[nti]}"
                if v in aset and v not in visited:
                    stack.append(v)
    return regions


def lewis_winnowing_attractor_analysis(
    duration: Counter,
    large_drop_ratio: float = 0.5,
) -> dict:
    """Lewis et al. (1999) winnowing（与原文 procedure 一致）。

    - 每轮剔除当前集合中**观测值最低**的格（本实现：总停留时长最短；并列按状态名排序）。
    - 各轮 H_j 同上式；比例 p_k = H_k/H_1。
    - 吸引子：在比例序列上找**相对降幅** (p_k-p_{k+1})/p_k 最大的一步；若该降幅 ≥约 50%（Lewis），
      则吸引子为这一剔除步**完成之后仍保留的格**（原文：examine scree, value *after* the largest drop；
      表 5.1：比例从约 0.55 到约 0.09 的陡降后**剩余两格**为吸引子）。
    - 若不出现这样的大跌落：取**时长最高**的单格（对应原文无大跌落时以最终单格迭代/scree 落至 0 的解读）。
    - 实现细节：不参与「仅剩 1 格时 p→0」的降幅比较，以免伪 100% 陡降掩盖多格情形下的真实 scree。
    """
    cells = [st for st, d in duration.items() if float(d) > 0]
    if not cells:
        return {
            "attractor_states": [],
            "heterogeneity_1": 0.0,
            "large_drop_threshold": large_drop_ratio,
            "largest_relative_drop": 0.0,
            "scree_drop_from_step": None,
            "used_large_drop_criterion": False,
            "degenerate_uniform": False,
            "winnowing_trace": [],
        }

    cells_sorted_asc = sorted(cells, key=lambda st: (float(duration[st]), st))
    n = len(cells)
    h_list: list[float] = []
    trace: list[dict] = []
    for k in range(n):
        remaining = cells_sorted_asc[k:]
        durs = [float(duration[st]) for st in remaining]
        h = lewis_heterogeneity(durs)
        h_list.append(h)
        removed = cells_sorted_asc[k - 1] if k > 0 else None
        trace.append(
            {
                "removals": k,
                "n_cells": n - k,
                "removed_state": removed,
                "heterogeneity": round(h, 8),
            }
        )

    h0 = h_list[0]
    if h0 <= 1e-12:
        for row in trace:
            row["h_proportion"] = 1.0 if row["removals"] == 0 else 0.0
        best = max(cells, key=lambda st: (float(duration[st]), st))
        return {
            "attractor_states": [best],
            "heterogeneity_1": round(h0, 8),
            "large_drop_threshold": large_drop_ratio,
            "largest_relative_drop": 0.0,
            "scree_drop_from_step": None,
            "used_large_drop_criterion": False,
            "degenerate_uniform": True,
            "winnowing_trace": trace,
        }

    p_list = [h_list[k] / h0 for k in range(n)]
    for k, row in enumerate(trace):
        row["h_proportion"] = round(p_list[k], 6)

    # 不参与「仅剩 1 格时 p→0」的跌落，否则相对跌落恒为 100%，与 Lewis 表 5.1 类 scree（仍多格）不符。
    best_drop = -1.0
    best_s = 0
    for s in range(n - 1):
        if n - (s + 1) < 2:
            continue
        if p_list[s] <= 1e-15:
            continue
        rel = (p_list[s] - p_list[s + 1]) / p_list[s]
        if rel > best_drop + 1e-12 or (abs(rel - best_drop) <= 1e-12 and s < best_s):
            best_drop = rel
            best_s = s

    use_large = best_drop >= large_drop_ratio - 1e-9
    if use_large:
        attractors = cells_sorted_asc[best_s + 1 :]
    else:
        attractors = [cells_sorted_asc[-1]]

    return {
        "attractor_states": attractors,
        "heterogeneity_1": round(h0, 8),
        "large_drop_threshold": large_drop_ratio,
        "largest_relative_drop": round(max(best_drop, 0.0), 6),
        "scree_drop_from_step": int(best_s) if n > 1 else None,
        "used_large_drop_criterion": bool(use_large),
        "degenerate_uniform": False,
        "winnowing_trace": trace,
    }


def interpret_level_by_ratio(ratio: float, low: float, high: float) -> str:
    if ratio < low:
        return "低"
    if ratio < high:
        return "中"
    return "高"


def duration_dispersion(
    duration: Counter,
    student_labels: list[str],
    teacher_labels: list[str],
) -> float:
    """
    分散度（SSG）：1 - ((n * Σ(d_i/D)²) - 1) / (n - 1)。
    D 为系统总持续时间，d_i 为第 i 格持续时间，n 为网格总单元格数；未访问格 d_i=0。
    全集中于一格 → 0；在全部 n 格均匀分布 → 1。
    """
    n = len(student_labels) * len(teacher_labels)
    if n <= 1:
        return 0.0
    d_total = float(sum(duration.values()))
    if d_total <= 0:
        return 0.0
    sum_p2 = 0.0
    for s in student_labels:
        for t in teacher_labels:
            key = f"{s} | {t}"
            di = float(duration.get(key, 0.0))
            p = di / d_total
            sum_p2 += p * p
    raw = 1.0 - ((n * sum_p2) - 1.0) / (n - 1.0)
    return max(0.0, min(1.0, raw))


def build_variability_analysis_lines(meta: dict) -> list[str]:
    """基础描述性分析：整合为单段描述。"""
    lines: list[str] = []
    m = meta
    cell_range_f = float(m.get("cell_range", 0))
    dispersion = float(m.get("dispersion", 0.0))
    ve_ratio = float(m.get("visits_events_ratio", 0.0))
    stickiness = float(m.get("stickiness", 0.0))

    disp_level = interpret_level_by_ratio(dispersion, 0.25, 0.6)
    sticky_level = "高" if stickiness >= 0.35 else ("中" if stickiness >= 0.15 else "低")
    if ve_ratio >= 0.85:
        repeat_note = "重复的师生互动状态较少"
    elif ve_ratio >= 0.65:
        repeat_note = "存在一定重复的师生互动状态，重复度中等"
    else:
        repeat_note = "同一格内重复的师生互动状态较多，重复度较高"

    lines.append(
        f"- 本课在固定 108 格状态空间中访问了 {int(cell_range_f)} 格；"
        f"分散度 Dispersion={dispersion:.3f}（{disp_level}），"
        f"表示各格时长占比在整网上的分布{'相对均匀、时间分散度较高' if disp_level != '低' else '较集中于少数格、时间分散度较低'}；"
        f"轨迹共 {m.get('trajectory_points', 0)} 个事件、{m.get('total_visits', 0)} 次访问与 {m.get('total_transitions', 0)} 次转换；"
        f"访问/事件比={ve_ratio:.3f}（{repeat_note}），粘滞度={stickiness:.3f}（{sticky_level}粘滞）。"
    )

    return lines


def build_markdown(task_id: str, diag: dict) -> str:
    lines = []
    lines.append("# 状态空间网格分析报告")
    lines.append("")
    lines.append(f"- **课程编号**：`{task_id}`")
    lines.append("- **师生互动状态空间网格坐标说明**：")
    lines.append(f"  x轴（学生认知投入）：{', '.join(FIXED_STUDENT_CATEGORIES)}")
    lines.append(f"  y轴（教师言语策略）：{', '.join(FIXED_TEACHER_CATEGORIES)}")
    lines.append("")
    lines.append("## 一、师生互动")
    lines.append("")
    lines.append("## 1. 基础描述性分析")
    lines.extend(build_variability_analysis_lines(diag["meta"]))
    lines.append("")
    lines.append("## 2. 变异性分析（熵指标）")
    lines.append("")
    eo = diag.get("entropy", {}).get("overall", {})
    hv = float(eo.get("visited_entropy", 0.0))
    hd = float(eo.get("duration_entropy", 0.0))
    ht = float(eo.get("transition_entropy", 0.0))
    hv_level = eo.get("visited_entropy_level", "低")
    hd_level = eo.get("duration_entropy_level", "低")
    ht_level = eo.get("transition_entropy_level", "低")
    if eo.get("visited_entropy_level_method") == "corpus_quartiles_raw_entropy":
        vr = eo.get("visited_entropy_level_quartiles_ref") or {}
        lines.append(
            "- **熵等级（低/中/高）**：对访问熵、持续时间熵、转换熵分别用 `tasks` 下**其它课程**已生成诊断中的**同类原始熵**计算全库 "
            f"Q1、Q2（中位数）、Q3；划分规则为 **低**：≤Q1；**中**：Q1<·≤Q2；**高**：>Q2（含第三、四四分位）。"
            f"当前访问熵参照库样本约 **n={vr.get('n_corpus_tasks', '')}** 课。"
        )
    else:
        lines.append(
            "- **熵等级（低/中/高）**：全库可参照课程不足 "
            f"（需≥{_MIN_CORPUS_TASKS_FOR_QUARTILE_LEVELS} 门其它课且各有 ssg_diagnosis.json）时，"
            "退化为归一化熵比例的三等分：**低**<0.33，**中**<0.67，否则**高**。"
        )
    lines.append("")
    lines.append(
        f"- 本课整体访问熵为{hv:.6f}（{hv_level}），说明对不同师生互动状态的访问分布"
        f"{'较为均衡、类型较丰富' if hv_level != '低' else '较集中于少数状态'}；"
        f"整体持续时间熵为{hd:.6f}（{hd_level}），表明课堂时间在各状态间的分配"
        f"{'相对均衡' if hd_level != '低' else '更偏向少数状态'}；"
        f"整体转换熵为{ht:.6f}（{ht_level}），反映以当前状态为条件时下一状态的不确定性"
        f"{'较高，互动不可预测性较强' if ht_level != '低' else '较低，互动路径较固定'}。"
    )
    lines.append("")
    lines.append("## 3. 吸引子分析")
    lines.append("")
    lw = diag.get("lewis_winnowing", {})
    n_reg = diag.get("attractor_region_component_count", 0)
    lines.append(
        f"- 本课异质性 $H_1$={lw.get('heterogeneity_1', '')}；最大相对跌落比例={lw.get('largest_relative_drop', '')} "
        f"（大跌落阈值={lw.get('large_drop_threshold', 0.5)}）；"
        f"是否采用大跌落判据={'是' if lw.get('used_large_drop_criterion') else '否'}；"
        f"退化（$H_1\\approx 0$ 近似均匀）={'是' if lw.get('degenerate_uniform') else '否'}。"
    )
    lines.append(
        f"- 共识别 **{len(diag.get('attractors', []))}** 个吸引子格；在 9×12 上网格 **四邻接** 连通片数：**{n_reg}**。"
        "（Lewis：多格若**相邻**可解读为同一吸引子**区域**，**不相邻**则多个独立吸引子。）"
    )
    lines.append("")
    if diag["attractors"]:
        for i, a in enumerate(diag["attractors"], start=1):
            student_state, teacher_state = a["state"].split(" | ", 1) if " | " in a["state"] else (a["state"], "")
            lines.append(
                f"{i}. 教师策略：{teacher_state}；学生投入：{student_state}。"
                f"该状态共被访问 {a['visits']} 次，转换 {a['transitions']} 次，平均每次停留 {a['duration per visit']} 秒。"
            )
    else:
        lines.append("- 无明显吸引子")
    lines.append("")
    lines.append("## 4. 师生耦合方向")
    lines.append("")
    cp = diag["coupling"]
    lines.append(
        f"- 同步变化={cp['synchronous']}，教师先行={cp['teacher_leads']}，学生先行={cp['student_leads']}，不明确={cp['ambiguous']}"
    )
    lines.append(f"- 主导模式：`{cp['dominant_mode']}`")
    lines.append("")
    lines.append("## 二、结合移动极值图")
    lines.append("")
    lines.append("## 5. 突然跃迁与异常点")
    lines.append("")
    js = diag.get("sliding_insights_source", diag.get("sudden_jumps_source", ""))
    lines.append(
        "**§5–§8 说明**：以下仅聚合 **教师言语策略**（出现频次、师侧策略变更 `t0→t1`），"
        "不展开学生认知标签，仅看教师侧聚合；"
        "变化也可能部分来自学生侧，需结合 **§4** 耦合与录像复核。"
    )
    lines.append("")
    lines.append(
        f"- **跃迁识别**：与 `j_sliding_report` 一致，对相邻 3 分钟滑动窗口 **移动均值** 的一阶差分按 |梯度| 高分位检出；"
        f"数据来自 **`{js}`**（侧车缺失时为同算法 **现场计算**）。"
    )
    lines.append("")
    if diag["critical_points"]:
        for c in diag["critical_points"]:
            d = c.get("direction", "")
            zh = "上升" if d == "up" else ("下降" if d == "down" else str(d))
            near = c.get("nearby_transitions", [])[:4]
            near_txt = near if near else "（该分钟邻域内无师侧策略变更）"
            lines.append(
                f"- 第{c['minute']}分钟附近 | 移动均值陡峭{zh} | 邻近 **师侧**变更={near_txt}"
            )
    else:
        lines.append("- 未检出显著的移动均值梯度跃迁（与滑动窗口报告一致）。")
    lines.append("")

    bridge = diag.get("sliding_ssg_bridge") or {}
    lines.append("## 7. 平台期识别与教师言语策略对照")
    lines.append("")
    lines.append(
        "- **滑动侧**：`sliding_insights.json` 的 `plateaus`（移动均值近平台）。"
        "**教师侧**：group 中点分钟落入该区间的轨迹上，统计 **不同教师策略种数**、**策略出现 Top**、"
        "落入区间的边上 **师侧变更次数** 及 **Top 师侧转移**。"
    )
    lines.append("")
    plat_rows = bridge.get("plateaus") or []
    if plat_rows:
        for i, row in enumerate(plat_rows, start=1):
            sp = row.get("span") or {}
            sg = row.get("ssg") or {}
            m0, m1 = sp.get("start_minute", ""), sp.get("end_minute", "")
            u = int(sg.get("unique_teachers", 0))
            hint = (
                "教师策略种类较少、头部策略集中，与宏观平台期下「教学支架相对稳定」较一致。"
                if u <= 3
                else "教师策略种类仍较多或轮换频繁，宏观均值虽平，师侧仍在多策略间切换。"
            )
            lines.append(
                f"- **平台期 {i}**（约 **{m0}–{m1} 分钟**）：轨迹事件 **{sg.get('n_events', 0)}** 条，"
                f"不同教师策略 **{u}** 种；落入区间边 **{sg.get('n_edges_into_span', 0)}** 条，"
                f"其中师侧变更 **{sg.get('n_teacher_shifts_into_span', 0)}** 次。"
                f" 策略 Top：{_fmt_ssg_top_nodes(sg.get('top_teachers') or [])}；"
                f"师侧转移 Top：{_fmt_teacher_trans_top(sg.get('top_teacher_transitions') or [])}。"
                f"{hint}"
            )
    else:
        lines.append("- 无平台期条目（滑动侧未检出或侧车无 `plateaus`）。")
    lines.append("")

    lines.append("## 7. 强/弱吸引子时段与教师言语策略对照")
    lines.append("")
    lines.append(
        "- **说明**：「强/弱吸引子」指 **滑动极值图** 上窄带 + 均值相对稳定（非 Lewis 格吸引子）。"
        "下表仅看 **教师策略** 分布与师侧变更。"
    )
    lines.append("")
    for label, key in (("强吸引子时段", "strong"), ("弱吸引子时段", "weak")):
        rows_b = bridge.get(key) or []
        lines.append(f"- **{label}**（滑动 `sliding_insights`）：")
        if rows_b:
            for i, row in enumerate(rows_b[:8], start=1):
                sp = row.get("span") or {}
                sg = row.get("ssg") or {}
                m0, m1 = sp.get("start_minute", ""), sp.get("end_minute", "")
                lines.append(
                    f"  - 片段 **{i}** **{m0}–{m1} 分钟**：事件 {sg.get('n_events', 0)}，"
                    f"教师策略 **{sg.get('unique_teachers', 0)}** 种，师侧变更 **{sg.get('n_teacher_shifts_into_span', 0)}** 次；"
                    f"策略 Top {_fmt_ssg_top_nodes(sg.get('top_teachers') or [])}；"
                    f"转移 Top {_fmt_teacher_trans_top(sg.get('top_teacher_transitions') or [])}"
                )
        else:
            lines.append("  - （无）")
    lines.append("")

    lines.append("## 8. 相变临界点与教师言语策略对照")
    lines.append("")
    lines.append(
        "- **滑动侧**：`phase_transitions` 为「窄带 A → 宽带鼓包 B → 窄带 C」；**临界点** 为 B 段起止分钟。"
        "以下对比 A/B/C 三段的 **教师策略种数**、**策略 Top** 与 **师侧转移 Top**。"
    )
    lines.append("")
    ph_rows = bridge.get("phases") or []
    if ph_rows:
        for i, row in enumerate(ph_rows, start=1):
            ph = row.get("phase") or {}
            sa, sb, sc = row.get("ssg_a") or {}, row.get("ssg_b") or {}, row.get("ssg_c") or {}
            a, b, c = ph.get("a_span") or {}, ph.get("b_span") or {}, ph.get("c_span") or {}
            dire = ph.get("direction", "")
            dzh = "向上" if dire == "up" else ("向下" if dire == "down" else str(dire))
            cm = ph.get("critical_minutes") or []
            lines.append(f"### 相变序列 {i}（**{dzh}**）")
            lines.append("")
            lines.append(
                f"- **段 A** {a.get('start_minute', '')}–{a.get('end_minute', '')} 分：教师策略 **{sa.get('unique_teachers', 0)}** 种，"
                f"Top {_fmt_ssg_top_nodes(sa.get('top_teachers') or [])}；"
                f"师侧转移 Top {_fmt_teacher_trans_top(sa.get('top_teacher_transitions') or [])}"
            )
            lines.append(
                f"- **段 B（鼓包）** {b.get('start_minute', '')}–{b.get('end_minute', '')} 分：教师策略 **{sb.get('unique_teachers', 0)}** 种，"
                f"Top {_fmt_ssg_top_nodes(sb.get('top_teachers') or [])}；"
                f"师侧转移 Top {_fmt_teacher_trans_top(sb.get('top_teacher_transitions') or [])}"
            )
            lines.append(
                f"- **段 C** {c.get('start_minute', '')}–{c.get('end_minute', '')} 分：教师策略 **{sc.get('unique_teachers', 0)}** 种，"
                f"Top {_fmt_ssg_top_nodes(sc.get('top_teachers') or [])}；"
                f"师侧转移 Top {_fmt_teacher_trans_top(sc.get('top_teacher_transitions') or [])}"
            )
            if len(cm) >= 2:
                lines.append(f"- **临界点（鼓包起止）**：**{cm[0]} 分钟**、**{cm[1]} 分钟**。")
            b_up = sb.get("unique_teachers", 0) > max(
                sa.get("unique_teachers", 0), sc.get("unique_teachers", 0)
            )
            lines.append(
                "- **解读提示**："
                + (
                    "鼓包段 **教师策略种类** 高于 A/C，与「师侧支架/活动类型切换更活跃」一致，可对照录像看是否换任务或换互动结构。"
                    if b_up
                    else "鼓包段教师策略种类未明显高于两侧，带宽鼓包可能更多反映学生侧离散或其它因素，需结合 **§4** 耦合与录像。"
                )
            )
            lines.append("")
    else:
        lines.append("- 未检出完整三相相变序列（与滑动报告第四节可能一致）。")
        lines.append("")

    lines.append("## 9. 教学解释建议")
    lines.append("")
    lines.append("- 优先检查 **§3 Lewis 吸引子格** 是否对应高质量认知目标；**§7** 滑动轨窄带时段再结合 **§6–§8** 看教师策略是否稳定或换挡。")
    lines.append("- 若 **§5 跃迁点** 邻域出现密集 **师侧策略变更**，可作为「可回看录像」的干预候选时刻。")
    lines.append("- 将 **§5–§8**（教师维度）与 `sliding/sliding_report.md` 对照，避免单维误读。")
    lines.append("")
    return "\n".join(lines)


def build_csv_rows(diag: dict) -> list[dict]:
    rows = []
    meta = diag.get("meta", {})
    entropy_overall = diag.get("entropy", {}).get("overall", {})
    rows.append(
        {
            "section": "meta",
            "item": "trajectory_points",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("trajectory_points", ""),
            "value2": "",
            "value3": "",
            "note": "",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "dispersion",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("dispersion", ""),
            "value2": "",
            "value3": "",
            "note": "1 - ((n*sum(d_i/D)^2)-1)/(n-1), n=108",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "total_duration_sec",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("total_duration_sec", ""),
            "value2": "",
            "value3": "",
            "note": "D in dispersion formula",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "visits_events_ratio",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("visits_events_ratio", ""),
            "value2": "",
            "value3": "",
            "note": "",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "stickiness",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("stickiness", ""),
            "value2": "",
            "value3": "",
            "note": "1 - visits/events",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "cell_range",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("cell_range", ""),
            "value2": "",
            "value3": "",
            "note": "",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "total_visits",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": meta.get("total_visits", ""),
            "value2": "",
            "value3": "",
            "note": "",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "visited_entropy",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": entropy_overall.get("visited_entropy", ""),
            "value2": entropy_overall.get("visited_entropy_norm", ""),
            "value3": "",
            "note": (
                f"level={entropy_overall.get('visited_entropy_level', '')}; "
                f"method={entropy_overall.get('visited_entropy_level_method', '')}"
            ),
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "duration_entropy",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": entropy_overall.get("duration_entropy", ""),
            "value2": entropy_overall.get("duration_entropy_norm", ""),
            "value3": "",
            "note": f"level={entropy_overall.get('duration_entropy_level', '')}",
        }
    )
    rows.append(
        {
            "section": "meta",
            "item": "transition_entropy",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": entropy_overall.get("transition_entropy", ""),
            "value2": entropy_overall.get("transition_entropy_norm", ""),
            "value3": "",
            "note": (
                f"level={entropy_overall.get('transition_entropy_level', '')}; "
                "H(Y|X) conditional ln; norm/ln(108)"
            ),
        }
    )

    for a in diag.get("attractors", []):
        rows.append(
            {
                "section": "attractor",
                "item": "state",
                "state": a.get("state", ""),
                "from_state": "",
                "to_state": "",
                "minute": "",
                "shape": "",
                "value1": a.get("visits", ""),
                "value2": a.get("events", ""),
                "value3": a.get("attractor_region_id", ""),
                "note": (
                    f"dur/visit={a.get('duration per visit', '')}; transitions={a.get('transitions', '')}; "
                    f"region_id={a.get('attractor_region_id', '')}"
                ),
            }
        )

    for c in diag.get("critical_points", []):
        rows.append(
            {
                "section": "critical_point",
                "item": "sudden_jump",
                "state": "",
                "from_state": "",
                "to_state": "",
                "minute": c.get("minute", ""),
                "shape": c.get("direction", ""),
                "value1": c.get("jumps_source", ""),
                "value2": "",
                "value3": "",
                "note": " | ".join(c.get("nearby_transitions", [])[:3]),
            }
        )

    coupling = diag.get("coupling", {})
    for key in ("synchronous", "teacher_leads", "student_leads", "ambiguous"):
        rows.append(
            {
                "section": "coupling",
                "item": key,
                "state": "",
                "from_state": "",
                "to_state": "",
                "minute": "",
                "shape": "",
                "value1": coupling.get(key, ""),
                "value2": "",
                "value3": "",
                "note": "",
            }
        )
    rows.append(
        {
            "section": "coupling",
            "item": "dominant_mode",
            "state": "",
            "from_state": "",
            "to_state": "",
            "minute": "",
            "shape": "",
            "value1": coupling.get("dominant_mode", ""),
            "value2": "",
            "value3": "",
            "note": "",
        }
    )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_split_tables(diag: dict) -> dict[str, tuple[list[str], list[dict]]]:
    eo = diag.get("entropy", {}).get("overall", {})
    meta_rows = [
        {"metric": "events", "value": diag.get("meta", {}).get("trajectory_points", "")},
        {"metric": "total_visits", "value": diag.get("meta", {}).get("total_visits", "")},
        {"metric": "cell_range", "value": diag.get("meta", {}).get("cell_range", "")},
        {"metric": "dispersion", "value": diag.get("meta", {}).get("dispersion", "")},
        {"metric": "total_duration_sec", "value": diag.get("meta", {}).get("total_duration_sec", "")},
        {"metric": "visits_events_ratio", "value": diag.get("meta", {}).get("visits_events_ratio", "")},
        {"metric": "stickiness", "value": diag.get("meta", {}).get("stickiness", "")},
        {"metric": "visited_entropy", "value": eo.get("visited_entropy", "")},
        {"metric": "duration_entropy", "value": eo.get("duration_entropy", "")},
        {"metric": "transition_entropy", "value": eo.get("transition_entropy", "")},
        {"metric": "visited_entropy_norm", "value": eo.get("visited_entropy_norm", "")},
        {"metric": "duration_entropy_norm", "value": eo.get("duration_entropy_norm", "")},
        {"metric": "transition_entropy_norm", "value": eo.get("transition_entropy_norm", "")},
    ]

    attractor_rows = []
    for a in diag.get("attractors", []):
        attractor_rows.append(
            {
                "state": a.get("state", ""),
                "visits": a.get("visits", ""),
                "events": a.get("events", ""),
                "total duration": a.get("total duration", ""),
                "duration per visit": a.get("duration per visit", ""),
                "transitions": a.get("transitions", ""),
                "attractor_region_id": a.get("attractor_region_id", ""),
            }
        )

    critical_rows = []
    for c in diag.get("critical_points", []):
        critical_rows.append(
            {
                "minute": c.get("minute", ""),
                "direction": c.get("direction", ""),
                "jumps_source": c.get("jumps_source", ""),
                "nearby_transitions": " | ".join(c.get("nearby_transitions", [])),
            }
        )

    cp = diag.get("coupling", {})
    coupling_rows = [
        {"metric": "synchronous", "value": cp.get("synchronous", "")},
        {"metric": "teacher_leads", "value": cp.get("teacher_leads", "")},
        {"metric": "student_leads", "value": cp.get("student_leads", "")},
        {"metric": "ambiguous", "value": cp.get("ambiguous", "")},
        {"metric": "dominant_mode", "value": cp.get("dominant_mode", "")},
    ]

    return {
        "ssg_meta.csv": (["metric", "value"], meta_rows),
        "ssg_attractors.csv": (
            [
                "state",
                "visits",
                "events",
                "total duration",
                "duration per visit",
                "transitions",
                "attractor_region_id",
            ],
            attractor_rows,
        ),
        "ssg_critical_points.csv": (
            ["minute", "direction", "jumps_source", "nearby_transitions"],
            critical_rows,
        ),
        "ssg_coupling.csv": (["metric", "value"], coupling_rows),
    }


def main():
    task_id = resolve_task_id()
    task_dir = Path("tasks") / task_id
    class_path = task_dir / "class" / "full.json"
    group_path = task_dir / "group" / "full.json"
    sliding_path = task_dir / "analysis" / "sliding_scores.json"

    class_data = load_json(class_path)
    group_data = load_json(group_path)
    sliding_data = load_json(sliding_path)

    bounds, _ = load_group_bounds(group_data)
    results = class_data.get("results", [])

    seq = []
    for r in results:
        gid = int(r["group"])
        if gid not in bounds:
            continue
        s = r["student"]["result"]
        t = normalize_teacher_result(r["teacher"]["result"])
        seq.append((gid, s, t))
    seq.sort(key=lambda x: x[0])

    if not seq:
        print("No valid trajectory points after alignment.")
        sys.exit(1)

    durations = {}
    for gid, _, _ in seq:
        t0, t1 = bounds[gid]
        durations[gid] = max(t1 - t0, 1.0)

    visits = Counter()
    events = Counter()
    duration = Counter()
    prev_state = None
    for gid, s, t in seq:
        state = f"{s} | {t}"
        events[state] += 1
        if state != prev_state:
            visits[state] += 1
        prev_state = state
        duration[state] += durations[gid]

    transitions = {}
    for state in visits:
        transitions[state] = max(visits[state] - 1, 0)

    duration_per_visit = {st: duration[st] / visits[st] for st in visits}

    lewis = lewis_winnowing_attractor_analysis(duration, large_drop_ratio=0.5)
    attractor_states = lewis["attractor_states"]
    region_map = attractor_region_ids_4neighbors(attractor_states)
    n_regions = len({region_map[s] for s in attractor_states if s in region_map})

    attractors = []
    for st in sorted(
        attractor_states,
        key=lambda x: (float(duration[x]), visits[x], x),
        reverse=True,
    ):
        attractors.append(
            {
                "state": st,
                "visits": int(visits[st]),
                "events": int(events[st]),
                "total duration": round(duration[st], 2),
                "duration per visit": round(duration_per_visit[st], 2),
                "transitions": int(transitions[st]),
                "attractor_region_id": int(region_map.get(st, 0)),
            }
        )

    transition_counts = Counter()
    for i in range(1, len(seq)):
        _, s0, t0 = seq[i - 1]
        _, s1, t1 = seq[i]
        from_state = f"{s0} | {t0}"
        to_state = f"{s1} | {t1}"
        k = (from_state, to_state)
        transition_counts[k] += 1

    tasks_root = Path("tasks")
    corp_ent = gather_corpus_entropy_from_tasks(tasks_root, exclude_task_id=task_id)
    corpus_entropy_quartiles: dict[str, tuple[float, float, float]] = {}
    corpus_entropy_sample_sizes: dict[str, int] = {}
    for mk in ("visited_entropy", "duration_entropy", "transition_entropy"):
        q123 = entropy_quartiles_q123(corp_ent[mk])
        if q123 is not None:
            corpus_entropy_quartiles[mk] = q123
            corpus_entropy_sample_sizes[mk] = len(corp_ent[mk])

    entropy_metrics = build_entropy_metrics(
        visits,
        duration,
        transition_counts,
        corpus_entropy_quartiles=corpus_entropy_quartiles or None,
        corpus_entropy_sample_sizes=corpus_entropy_sample_sizes or None,
    )
    teacher_bins, student_bins, bin_size_sec = build_binned_teacher_student_sequences(seq, bounds)
    transition_analysis = build_cross_variable_transition_analysis(teacher_bins, student_bins)
    transition_analysis["bin_size_sec"] = round(bin_size_sec, 6)

    dispersion_val = duration_dispersion(duration, FIXED_STUDENT_CATEGORIES, FIXED_TEACHER_CATEGORIES)

    group_to_minute = {}
    for gid, (t0, t1) in bounds.items():
        mid = (t0 + t1) / 2.0
        group_to_minute[gid] = int(mid // 60) + 1

    windows_3 = sliding_data.get("windows_3min", [])
    sliding_insights, sliding_src = load_sliding_insights(task_dir, windows_3)
    jump_list = sliding_insights.get("jumps", [])
    if not isinstance(jump_list, list):
        jump_list = []
    jump_list = _normalize_jump_items(jump_list)
    trans_teacher_by_minute = defaultdict(list)
    for i in range(1, len(seq)):
        gid_cur = seq[i][0]
        _, _s0, t0 = seq[i - 1]
        _, _s1, t1 = seq[i]
        m = group_to_minute.get(gid_cur)
        if m is None:
            continue
        if t0 != t1:
            trans_teacher_by_minute[m].append(f"{t0} -> {t1}")

    critical_points = []
    for j in jump_list:
        m = j["minute"]
        nearby: list[str] = []
        for mm in (m - 1, m, m + 1):
            nearby.extend(trans_teacher_by_minute.get(mm, []))
        critical_points.append(
            {
                "minute": m,
                "direction": j["direction"],
                "nearby_transitions": nearby[:6],
                "jumps_source": sliding_src,
            }
        )

    sliding_ssg_bridge = build_sliding_ssg_bridge(seq, group_to_minute, sliding_insights)

    coupling = coupling_direction(seq)

    diagnosis = {
        "meta": {
            "task_id": task_id,
            "trajectory_points": len(seq),
            "total_visits": int(sum(visits.values())),
            "total_transitions": int(max(sum(visits.values()) - 1, 0)),
            "cell_range": len(visits),
            "dispersion": round(dispersion_val, 6),
            "total_duration_sec": round(float(sum(duration.values())), 4),
            "visits_events_ratio": round((sum(visits.values()) / len(seq)) if len(seq) > 0 else 0.0, 6),
            "stickiness": round((1.0 - (sum(visits.values()) / len(seq))) if len(seq) > 0 else 0.0, 6),
        },
        "entropy": entropy_metrics,
        "transition_analysis": transition_analysis,
        "attractor_methodology": {
            "source": "Lewis et al. (1999), first state space grid application — winnowing method",
            "null_hypothesis": (
                "Behavior equally distributed across cells; expected duration per cell = "
                "total duration in the current set / number of cells (chi-square style)."
            ),
            "observed": "per-cell total duration (seconds); visited cells only (duration > 0)",
            "removal_rule": "each iteration remove the cell with lowest total duration (ties: lexicographic state)",
            "heterogeneity_j": (
                "H_j = (1/n_j) * sum_i (O_i - E_j)^2 / E_j; E_j = total_duration_in_step / n_j"
            ),
            "scree_lewis_1999": (
                "Quantify h_proportion_k = H_k / H_1; find the largest proportional drop "
                "( (p_k - p_{k+1}) / p_k ). 'Large' drop ~>= 50% (Lewis et al.). "
                "Attractors = cells remaining AFTER that drop (paper: value after the largest drop; "
                "e.g. Table 5.1 two cells qualify). "
                "Implementation excludes the 2-cell->1-cell step where p falls to 0 to avoid a spurious 100% drop. "
                "If no large drop: single attractor = cell with highest total duration "
                "(paper: scree to last single-cell iteration / drop to 0)."
            ),
            "regions_lewis": (
                "Multiple attractor cells adjacent => one attractor region; non-adjacent => multiple attractors. "
                "Encoded via 4-neighbor connected components on the fixed 9x12 grid (attractor_region_id)."
            ),
        },
        "lewis_winnowing": lewis,
        "attractor_region_component_count": n_regions,
        "attractors": attractors,
        "critical_points": critical_points,
        "sudden_jumps_source": sliding_src,
        "sliding_insights": sliding_insights,
        "sliding_insights_source": sliding_src,
        "sliding_ssg_bridge": sliding_ssg_bridge,
        "coupling": coupling,
    }

    out_dir = task_dir / "ssg"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ssg_diagnosis.json"
    md_path = out_dir / "ssg_diagnosis.md"
    json_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(task_id, diagnosis), encoding="utf-8")
    split_tables = build_split_tables(diagnosis)
    for filename, (fieldnames, rows) in split_tables.items():
        write_csv(out_dir / filename, fieldnames, rows)

    print(f"Saved diagnosis JSON: {json_path}")
    print(f"Saved diagnosis Markdown: {md_path}")
    print("Saved split CSV tables: ssg_meta.csv, ssg_attractors.csv, ssg_critical_points.csv, ssg_coupling.csv")


if __name__ == "__main__":
    main()
