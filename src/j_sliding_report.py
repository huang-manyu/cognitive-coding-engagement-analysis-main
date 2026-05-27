"""
从 analysis/sliding_scores.json 生成单课中文 Markdown「分析报告」。
维度：趋势与水平、变异性、吸引子、相变临界点（窄→鼓包→窄且新水平）。
日志输出为英文。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

# 须与 i_analysis.SLIDING_WINDOW_MINUTES 一致
SLIDING_WINDOW_MINUTES = 3
# 突然跃迁识别（写入 sliding_insights.json，与 k_ssg_diagnosis 共用）
SLIDING_JUMP_QUANTILE = 0.88
SLIDING_JUMP_MAX_ITEMS = 8
# 强吸引子：合并为连续段后，覆盖分钟跨度须 ≥ 此值，排除转瞬即逝的偶发窄带
STRONG_ATTRACTOR_MIN_SPAN_MINUTES = 5


def resolve_task_ids(tasks_dir: Path, arg: str | None) -> list[str]:
    if arg:
        return [arg]
    out: list[str] = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "analysis" / "sliding_scores.json").is_file():
            out.append(d.name)
    return out


def max_lesson_end_seconds(task_dir: Path) -> float | None:
    group_path = task_dir / "group" / "full.json"
    if not group_path.is_file():
        return None
    data = json.loads(group_path.read_text(encoding="utf-8"))
    mx = 0.0
    for seg in data.get("segments", []):
        mx = max(mx, float(seg.get("end", 0.0)))
    return mx if mx > 0 else None


def window_label(minutes_1based: list[int]) -> str:
    a, b = minutes_1based[0], minutes_1based[-1]
    if a == b:
        return f"第{a}分钟"
    return f"第{a}–{b}分钟"


def score_str(x: float) -> str:
    """带宽等变异性指标用；报告中不输出认知投入度（移动均值）具体分值。"""
    t = f"{x:.3f}".rstrip("0").rstrip(".")
    return t if t else "0"


def mean_level_in_trio(m: float, m1: float, m2: float, m3: float, eps: float = 0.012) -> str:
    """三段均值相对高低定性描述，不出现数字。"""
    lo, hi = min(m1, m2, m3), max(m1, m2, m3)
    if hi - lo < eps:
        return "在本课前/中/后三段中 **与另两段接近**"
    if m <= lo + 1e-9:
        return "在本课前/中/后三段中 **相对最低**"
    if m >= hi - 1e-9:
        return "在本课前/中/后三段中 **相对最高**"
    return "在本课前/中/后三段中 **居中**"


def phase_mean_level_phrase(mean_a: float, mean_c: float, eps: float = 0.018) -> tuple[str, str]:
    """相变段 A / C 的均值定性（相对全课），不出现数字。"""
    lo, hi = min(mean_a, mean_c), max(mean_a, mean_c)
    if abs(mean_a - mean_c) < eps:
        a_phrase = "一段相对稳定区"
        c_phrase = "另一段相对稳定区（与段 A 水平有差异但幅度有限）"
        return a_phrase, c_phrase
    if mean_a < mean_c:
        return "相对较低的一段稳定区", "相对较高的一段稳定区"
    return "相对较高的一段稳定区", "相对较低的一段稳定区"


def percentile_linear(sorted_vals: list[float], p: float) -> float:
    """p in [0,1]."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    x = p * (n - 1)
    lo = int(x)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return sorted_vals[lo]
    t = x - lo
    return sorted_vals[lo] * (1 - t) + sorted_vals[hi] * t


def merge_minute_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    norm = sorted((min(a, b), max(a, b)) for a, b in spans)
    out: list[tuple[int, int]] = [norm[0]]
    for a, b in norm[1:]:
        la, lb = out[-1]
        if a <= lb + 1:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def window_range_to_minute_span(windows: list[dict], i0: int, i1: int) -> tuple[int, int]:
    """闭区间窗口下标 [i0, i1] 对应覆盖的分钟范围（起止分钟，1-based）。"""
    m0 = windows[i0]["minutes_1based"][0]
    m1 = windows[i1]["minutes_1based"][-1]
    return (m0, m1)


def idx_ranges_from_indices(idxs: list[int]) -> list[tuple[int, int]]:
    if not idxs:
        return []
    idxs = sorted(idxs)
    runs: list[tuple[int, int]] = []
    s = p = idxs[0]
    for j in idxs[1:]:
        if j == p + 1:
            p = j
        else:
            runs.append((s, p))
            s = p = j
    runs.append((s, p))
    return runs


def filter_strong_attractor_ranges_by_duration(
    windows: list[dict],
    ranges: list[tuple[int, int]],
    min_span_minutes: int = STRONG_ATTRACTOR_MIN_SPAN_MINUTES,
) -> list[tuple[int, int]]:
    """仅保留覆盖时长足够的强吸引子连续段（闭区间窗口下标 → 起止分钟跨度）。"""
    if min_span_minutes <= 0:
        return list(ranges)
    out: list[tuple[int, int]] = []
    for s, e in ranges:
        m0, m1 = window_range_to_minute_span(windows, s, e)
        if m1 - m0 + 1 >= min_span_minutes:
            out.append((s, e))
    return out


def format_minute_spans(spans: list[tuple[int, int]]) -> str:
    if not spans:
        return "本课数据中未稳定检出。"
    parts: list[str] = []
    for a, b in spans:
        if a == b:
            parts.append(f"{a}分钟")
        else:
            parts.append(f"{a}-{b}分钟")
    return "；".join(parts)


def format_window_index_ranges(windows: list[dict], ranges: list[tuple[int, int]]) -> str:
    spans: list[tuple[int, int]] = []
    for s, e in ranges:
        spans.append(window_range_to_minute_span(windows, s, e))
    return format_minute_spans(merge_minute_spans(spans))


def lesson_duration_line(end_sec: float | None, num_minutes: int) -> str:
    if end_sec is not None:
        m = end_sec / 60.0
        if abs(m - round(m)) < 0.05:
            return f"- **本课时长**：约 {int(round(m))} 分钟"
        return f"- **本课时长**：约 {m:.1f} 分钟"
    return f"- **本课时长**：约 {num_minutes} 分钟（未找到 `group/full.json`，按分钟槽数）"


def series_from_windows(windows: list[dict]) -> tuple[list[float], list[float], list[float], list[float]]:
    avgs = [float(w["avg"]) for w in windows]
    mins_v = [float(w["min"]) for w in windows]
    maxs_v = [float(w["max"]) for w in windows]
    bws = [maxs_v[i] - mins_v[i] for i in range(len(windows))]
    return avgs, mins_v, maxs_v, bws


def gradients(avgs: list[float]) -> list[float]:
    """相邻窗口移动均值之差（步长 1 分钟），长度 n-1。"""
    return [avgs[i + 1] - avgs[i] for i in range(len(avgs) - 1)]


def plateau_segments_by_slope(
    avgs: list[float],
    slopes: list[float],
    slope_flat_thr: float,
    level_eps: float,
    min_windows: int,
) -> list[tuple[int, int]]:
    """斜率接近 0 且段内水平起伏小 → 平台期（窗口下标闭区间）。"""
    n = len(avgs)
    if n < min_windows or not slopes:
        return []
    segs: list[tuple[int, int]] = []
    s = 0
    for e in range(1, n):
        if abs(slopes[e - 1]) > slope_flat_thr:
            if e - s >= min_windows:
                chunk = avgs[s:e]
                if max(chunk) - min(chunk) <= level_eps:
                    segs.append((s, e - 1))
            s = e
    if n - s >= min_windows:
        chunk = avgs[s:n]
        if max(chunk) - min(chunk) <= level_eps and all(
            abs(slopes[j]) <= slope_flat_thr for j in range(s, n - 1)
        ):
            segs.append((s, n - 1))
    return segs


def jump_points_from_slopes(
    windows: list[dict],
    slopes: list[float],
    quantile: float = 0.88,
    max_items: int = 8,
) -> list[tuple[int, str]]:
    """|斜率| 处于高分位 → 突然跃迁，返回 (分钟, up/down)。"""
    if not slopes:
        return []
    abs_s = sorted(abs(x) for x in slopes)
    thr = abs_s[max(0, min(len(abs_s) - 1, int((len(abs_s) - 1) * quantile)))]
    if thr <= 1e-9:
        return []
    out: list[tuple[int, float]] = []
    for i, g in enumerate(slopes):
        if abs(g) >= thr:
            out.append((i, g))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    events: list[tuple[int, str]] = []
    seen_m: set[int] = set()
    for i, g in out:
        if len(events) >= max_items:
            break
        anchor = windows[i + 1]["minutes_1based"][0]
        if anchor in seen_m:
            continue
        seen_m.add(anchor)
        direction = "up" if g > 0 else "down"
        events.append((anchor, direction))
    # 按时间顺序输出，满足用户“按时间顺序”的要求
    events.sort(key=lambda x: x[0])
    return events


def build_sliding_insights_document(windows: list[dict]) -> dict:
    """
    单一侧车 JSON：`sliding/sliding_insights.json`，供 k_ssg_diagnosis 等使用。
    含：突然跃迁、平台期、强/弱吸引子时段（滑动轨语义）、相变序列与临界点分钟。
    """
    empty = {
        "source": "j_sliding_report",
        "version": 1,
        "jump_quantile": SLIDING_JUMP_QUANTILE,
        "jump_max_items": SLIDING_JUMP_MAX_ITEMS,
        "jumps": [],
        "plateaus": [],
        "strong_attractor_spans": [],
        "weak_attractor_spans": [],
        "phase_transitions": [],
    }
    if not windows:
        return empty

    avgs, _, _, bws = series_from_windows(windows)
    n = len(windows)
    slopes = gradients(avgs)
    sorted_bw = sorted(bws)
    abs_sl = sorted(abs(x) for x in slopes) if slopes else [0.0]
    slope_flat = max(percentile_linear(abs_sl, 0.35), 0.004)
    plats = plateau_segments_by_slope(avgs, slopes, slope_flat, level_eps=0.012, min_windows=3)
    strong_r, weak_r = attractor_segments(windows, avgs, bws, slopes, sorted_bw)
    phases = find_phase_transitions(windows, avgs, bws)
    jump_pairs = jump_points_from_slopes(
        windows,
        slopes,
        quantile=SLIDING_JUMP_QUANTILE,
        max_items=SLIDING_JUMP_MAX_ITEMS,
    )

    def span_to_dict(s: int, e: int) -> dict[str, int]:
        m0, m1 = window_range_to_minute_span(windows, s, e)
        return {"start_minute": int(m0), "end_minute": int(m1)}

    phases_out: list[dict] = []
    for ph in phases[:5]:
        a0, a1 = ph.a_span
        b0, b1 = ph.b_span
        c0, c1 = ph.c_span
        higher = ph.mean_c > ph.mean_a
        phases_out.append(
            {
                "a_span": {"start_minute": int(a0), "end_minute": int(a1)},
                "b_span": {"start_minute": int(b0), "end_minute": int(b1)},
                "c_span": {"start_minute": int(c0), "end_minute": int(c1)},
                "direction": "up" if higher else "down",
                "critical_minutes": [int(b0), int(b1)],
            }
        )

    return {
        "source": "j_sliding_report",
        "version": 1,
        "jump_quantile": SLIDING_JUMP_QUANTILE,
        "jump_max_items": SLIDING_JUMP_MAX_ITEMS,
        "jumps": [{"minute": int(m), "direction": d} for m, d in jump_pairs],
        "plateaus": [span_to_dict(s, e) for s, e in plats[:12]],
        "strong_attractor_spans": [span_to_dict(s, e) for s, e in strong_r[:24]],
        "weak_attractor_spans": [span_to_dict(s, e) for s, e in weak_r[:24]],
        "phase_transitions": phases_out,
    }


def jump_cluster_text(events: list[tuple[int, str]]) -> str:
    """将跃迁点按方向聚类并生成中文叙述。"""
    if not events:
        return "未检出显著的梯度尖峰，移动均值以渐进变化为主。"
    up_minutes = [m for m, d in events if d == "up"]
    down_minutes = [m for m, d in events if d == "down"]

    def mins_to_text(ms: list[int]) -> str:
        if not ms:
            return ""
        return "、".join([f"第{m}分钟" for m in ms])

    parts: list[str] = ["课堂经历了多次显著的波动跃迁。"]
    if up_minutes:
        parts.append(f"其中在 {mins_to_text(up_minutes)} 前后出现了明显的认知投入度上升。")
    if down_minutes:
        parts.append(
            f"而在 {mins_to_text(down_minutes)} 前后出现断崖式下降，表明课堂教学节奏在这些节点发生了急剧转折。"
        )
    return "".join(parts)


def window_third_boundaries(n: int) -> tuple[int, int]:
    """将 n 个滑动窗口划为前/中/后三段，返回 (i1, i2)，段为 [0,i1)、[i1,i2)、[i2,n)。"""
    if n <= 0:
        return 0, 0
    if n < 3:
        return 1, min(2, n)
    i1 = n // 3
    i2 = 2 * n // 3
    if i1 < 1:
        i1 = 1
    if i2 <= i1:
        i2 = i1 + 1
    if i2 >= n:
        i2 = n - 1
    return i1, i2


def _three_segment_mean_qualitative(m1: float, m2: float, m3: float, eps: float = 0.012) -> str:
    """前/中/后三段均值高低的简短概括。"""
    lo, hi = min(m1, m2, m3), max(m1, m2, m3)
    if hi - lo < eps:
        return "三段均值接近，全课水平场整体较平稳。"
    if m2 > m1 + eps and m2 > m3 + eps:
        return "中段相对前、后段 **抬高**，呈中间凸起形态。"
    if m2 + eps < m1 and m2 + eps < m3:
        return "中段相对前、后段 **压低**，呈中间凹陷形态。"
    if m3 > m2 + eps and m3 > m1 + eps:
        return "由前段经中段至后段 **阶梯抬升**，后段投入水平最高。"
    if m1 > m2 + eps and m1 > m3 + eps:
        return "由前段经中段至后段 **阶梯回落**，前段投入水平最高。"
    if m3 > m1 + eps and abs(m2 - m1) < eps * 1.5:
        return "后段相对前段 **更高**，中段过渡。"
    if m1 > m3 + eps and abs(m2 - m3) < eps * 1.5:
        return "后段相对前段 **更低**，中段过渡。"
    return "前、中、后段水平 **交错起伏**，无简单单调形态。"


def overall_trend_narrative(
    avgs: list[float],
    bws: list[float],
    i_max: int,
    i_min: int,
    windows: list[dict],
) -> str:
    """按前段、中段、后段（滑动窗口三等分）概括移动均值，并补充全课带宽气质与高低点。"""
    n = len(avgs)
    vol = median(bws) if bws else 0.0
    hi_m = window_label(windows[i_max]["minutes_1based"])
    lo_m = window_label(windows[i_min]["minutes_1based"])
    if n < 3:
        parts = [f"滑动窗口仅 {n} 个，无法划分前/中/后三段；全课请直接对照折线整体走势。"]
        if vol > 0.12:
            parts.append(" 全课带宽中位数偏高，波动色彩较强。")
        elif vol < 0.05:
            parts.append(" 全课带宽整体偏窄。")
        else:
            parts.append(" 全课带宽中等。")
        parts.append(f" 移动均值相对高点在 **{hi_m}**，相对低点在 **{lo_m}**。")
        return "".join(parts)
    i1, i2 = window_third_boundaries(n)
    m1 = sum(avgs[0:i1]) / max(1, i1)
    m2 = sum(avgs[i1:i2]) / max(1, i2 - i1)
    m3 = sum(avgs[i2:n]) / max(1, n - i2)
    a0, b0 = window_range_to_minute_span(windows, 0, i1 - 1)
    a1, b1 = window_range_to_minute_span(windows, i1, i2 - 1)
    a2, b2 = window_range_to_minute_span(windows, i2, n - 1)

    parts: list[str] = []
    parts.append(
        f"**前段**（约 **{a0}–{b0} 分钟**），移动窗口均值{mean_level_in_trio(m1, m1, m2, m3)}；"
        f"**中段**（约 **{a1}–{b1} 分钟**），移动窗口均值{mean_level_in_trio(m2, m1, m2, m3)}；"
        f"**后段**（约 **{a2}–{b2} 分钟**），移动窗口均值{mean_level_in_trio(m3, m1, m2, m3)}。"
        f"{_three_segment_mean_qualitative(m1, m2, m3)}"
    )
    if vol > 0.12:
        parts.append(" 全课带宽中位数偏高，配合均值起伏，波动与探索色彩较强。")
    elif vol < 0.05:
        parts.append(" 全课带宽整体偏窄，群体输出相对一致。")
    else:
        parts.append(" 全课带宽中等，起伏与稳定段交替出现。")
    parts.append(f" 移动均值的相对高点在 **{hi_m}**，相对低点在 **{lo_m}**。")
    return "".join(parts)


def extreme_bw_ranges(
    bws: list[float],
    low_p: float = 0.22,
    high_p: float = 0.78,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float, float]:
    """极窄 / 极宽带宽窗口下标连续段。"""
    n = len(bws)
    if n == 0:
        return [], [], 0.0, 0.0
    sb = sorted(bws)
    thr_lo = percentile_linear(sb, low_p)
    thr_hi = percentile_linear(sb, high_p)
    narrow_i = [i for i in range(n) if bws[i] <= thr_lo]
    wide_i = [i for i in range(n) if bws[i] >= thr_hi]
    return idx_ranges_from_indices(narrow_i), idx_ranges_from_indices(wide_i), thr_lo, thr_hi


def bandwidth_three_segment_trend_text(windows: list[dict], bws: list[float]) -> str:
    """按前段、中段、后段（与均值三等分同一划界）描述平均带宽及段间对比。"""
    n = len(bws)
    if n < 3:
        return "窗口数量过少，不作前/中/后段带宽概括。"
    i1, i2 = window_third_boundaries(n)
    b1 = sum(bws[0:i1]) / max(1, i1)
    b2 = sum(bws[i1:i2]) / max(1, i2 - i1)
    b3 = sum(bws[i2:n]) / max(1, n - i2)
    m0a, m0b = window_range_to_minute_span(windows, 0, i1 - 1)
    m1a, m1b = window_range_to_minute_span(windows, i1, i2 - 1)
    m2a, m2b = window_range_to_minute_span(windows, i2, n - 1)

    head = (
        f"**前段**（约 **{m0a}–{m0b} 分钟**）平均带宽 **{score_str(b1)}**；"
        f"**中段**（约 **{m1a}–{m1b} 分钟**）**{score_str(b2)}**；"
        f"**后段**（约 **{m2a}–{m2b} 分钟**）**{score_str(b3)}**。"
    )
    r12 = (b2 - b1) / max(b1, 1e-6)
    r23 = (b3 - b2) / max(b2, 1e-6)
    r13 = (b3 - b1) / max(b1, 1e-6)
    if r12 > 0.12 and r23 < -0.12:
        tail = " 段间对比：带宽 **前段→中段扩张、中段→后段收窄**，后段变异性相对中段下降。"
    elif r12 < -0.12 and r23 > 0.12:
        tail = " 段间对比：带宽 **前段→中段收窄、中段→后段扩张**，后段离散度升高。"
    elif r12 > 0.12 and r23 > 0.12:
        tail = " 段间对比：**前段→中段→后段持续变宽**，变异性沿时间递增。"
    elif r12 < -0.12 and r23 < -0.12:
        tail = " 段间对比：**前段→中段→后段持续变窄**，系统逐步收敛。"
    elif r13 > 0.12:
        tail = " 段间对比：后段平均带宽 **高于** 前段，后段整体离散度更强。"
    elif r13 < -0.12:
        tail = " 段间对比：后段平均带宽 **低于** 前段，后段整体更趋收敛。"
    else:
        tail = " 段间对比：三段带宽 **无单调大幅升降**，呈交替或平缓变化。"
    return head + tail


def attractor_segments(
    windows: list[dict],
    avgs: list[float],
    bws: list[float],
    slopes: list[float],
    sorted_bw: list[float],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    强吸引子：带宽极低 + 段内均值平 + 相邻斜率小，且合并为连续段后覆盖分钟数足够长
    （非转瞬即逝的偶发窄带）。弱吸引子：带宽较低但高于强阈值、可见分离；均值小幅波动。
    """
    n = len(avgs)
    if n == 0:
        return [], []
    q10 = percentile_linear(sorted_bw, 0.10)
    q35 = percentile_linear(sorted_bw, 0.35)
    # 强：带宽不超过 q10 与绝对上限的较小者
    strong_thr = min(max(q10, 1e-6), 0.035)
    weak_hi = max(q35, strong_thr + 0.012)

    strong_idx: list[int] = []
    weak_idx: list[int] = []
    for i in range(n):
        bw = bws[i]
        g_lo = abs(slopes[i - 1]) if i > 0 else 0.0
        g_hi = abs(slopes[i]) if i < len(slopes) else 0.0
        loc_lo = avgs[i] - min(avgs[max(0, i - 1) : min(n, i + 2)])
        loc_hi = max(avgs[max(0, i - 1) : min(n, i + 2)]) - avgs[i]
        local_span = loc_lo + loc_hi
        if bw <= strong_thr and local_span <= 0.04 and max(g_lo, g_hi) <= 0.025:
            strong_idx.append(i)
        elif strong_thr < bw <= weak_hi and local_span <= 0.07 and max(g_lo, g_hi) <= 0.035:
            weak_idx.append(i)

    # 从弱中去掉已标强的邻域扩展（避免重复叙述）
    weak_idx = [i for i in weak_idx if i not in strong_idx]
    strong_r = idx_ranges_from_indices(strong_idx)
    strong_r = filter_strong_attractor_ranges_by_duration(windows, strong_r)
    return strong_r, idx_ranges_from_indices(weak_idx)


@dataclass
class PhaseTransition:
    a_span: tuple[int, int]
    b_span: tuple[int, int]
    c_span: tuple[int, int]
    mean_a: float
    mean_c: float
    mean_bw_a: float
    mean_bw_b: float
    mean_bw_c: float


def find_phase_transitions(
    windows: list[dict],
    avgs: list[float],
    bws: list[float],
    min_run_narrow: int = 2,
    min_run_bulge: int = 1,
    mean_diff_min: float = 0.018,
    bulge_ratio_min: float = 1.35,
) -> list[PhaseTransition]:
    """
    严格序列：A 窄带低方差 → B 带宽鼓包（相对 A/C 显著高）→ C 窄带且均值与 A 不同。
    临界点：B 段的起点分钟、终点分钟（覆盖窗口的分钟边界）。
    """
    n = len(windows)
    if n < min_run_narrow + min_run_bulge + min_run_narrow:
        return []

    sb = sorted(bws)
    # 窄：低于 30% 分位；鼓包：高于 65% 分位（同一课内相对）
    thr_narrow = percentile_linear(sb, 0.30)
    thr_wide = percentile_linear(sb, 0.65)

    def run_type(i: int) -> str:
        if bws[i] <= thr_narrow:
            return "N"
        if bws[i] >= thr_wide:
            return "W"
        return "M"

    labels = [run_type(i) for i in range(n)]
    # 拆成极大连续段，类型为 N/W/M
    runs: list[tuple[str, int, int]] = []
    s = 0
    for i in range(1, n):
        if labels[i] != labels[s]:
            runs.append((labels[s], s, i - 1))
            s = i
    runs.append((labels[s], s, n - 1))

    out: list[PhaseTransition] = []
    for ri in range(len(runs) - 2):
        t_a, a0, a1 = runs[ri]
        t_b, b0, b1 = runs[ri + 1]
        t_c, c0, c1 = runs[ri + 2]
        if t_a != "N" or t_b != "W" or t_c != "N":
            continue
        if a1 - a0 + 1 < min_run_narrow or b1 - b0 + 1 < min_run_bulge or c1 - c0 + 1 < min_run_narrow:
            continue
        mean_a = sum(avgs[a0 : a1 + 1]) / (a1 - a0 + 1)
        mean_c = sum(avgs[c0 : c1 + 1]) / (c1 - c0 + 1)
        if abs(mean_a - mean_c) < mean_diff_min:
            continue
        mean_bw_a = sum(bws[a0 : a1 + 1]) / (a1 - a0 + 1)
        mean_bw_b = sum(bws[b0 : b1 + 1]) / (b1 - b0 + 1)
        mean_bw_c = sum(bws[c0 : c1 + 1]) / (c1 - c0 + 1)
        if mean_bw_b < mean_bw_a * bulge_ratio_min or mean_bw_b < mean_bw_c * bulge_ratio_min:
            continue
        if mean_bw_a <= 1e-9 and mean_bw_b < 0.02:
            continue
        # C 须回归窄带：相对鼓包明显下降
        if mean_bw_c > mean_bw_b * 0.88:
            continue
        if mean_bw_c > max(mean_bw_a * 1.6, thr_narrow * 2.2):
            continue
        a_span = window_range_to_minute_span(windows, a0, a1)
        b_span = window_range_to_minute_span(windows, b0, b1)
        c_span = window_range_to_minute_span(windows, c0, c1)
        out.append(
            PhaseTransition(
                a_span=a_span,
                b_span=b_span,
                c_span=c_span,
                mean_a=mean_a,
                mean_c=mean_c,
                mean_bw_a=mean_bw_a,
                mean_bw_b=mean_bw_b,
                mean_bw_c=mean_bw_c,
            )
        )
    return out


def build_markdown(task_id: str, payload: dict, end_sec: float | None) -> str:
    num_minutes = int(payload.get("num_minutes", 0))
    windows: list[dict] = payload.get("windows_3min", [])
    lines: list[str] = []

    lines.append("# 学生认知投入分析报告")
    lines.append("")
    lines.append(f"- **课程编号**：`{task_id}`")
    lines.append(lesson_duration_line(end_sec, num_minutes))
    lines.append("")

    lines.append("## 方法说明")
    lines.append("")
    lines.append(
        f"本报告基于 **移动极值图**（{SLIDING_WINDOW_MINUTES} 分钟滑动窗口，步长 1 分钟）统计。"
        "图中 **移动均值**（折线中通常为中间走势）刻画群体认知投入平均水平；"
        "**极大值 / 极小值** 对应窗口内得分上下界，二者纵向间距即为 **带宽**。"
    )
    lines.append("")

    if not windows:
        lines.append("当前无滑动窗口数据，无法生成四维度分析。")
        return "\n".join(lines)

    avgs, _, _, bws = series_from_windows(windows)
    n = len(windows)
    slopes = gradients(avgs)
    sorted_bw = sorted(bws)

    i_max = max(range(n), key=lambda i: avgs[i])
    i_min = min(range(n), key=lambda i: avgs[i])

    # 斜率阈值：平台用分位，随课自适应
    abs_sl = sorted(abs(x) for x in slopes) if slopes else [0.0]
    slope_flat = percentile_linear(abs_sl, 0.35)
    slope_flat = max(slope_flat, 0.004)

    plats = plateau_segments_by_slope(avgs, slopes, slope_flat, level_eps=0.012, min_windows=3)
    jumps = jump_points_from_slopes(
        windows,
        slopes,
        quantile=SLIDING_JUMP_QUANTILE,
        max_items=SLIDING_JUMP_MAX_ITEMS,
    )

    lines.append("## 一、总体发展趋势")
    lines.append("")
    lines.append(overall_trend_narrative(avgs, bws, i_max, i_min, windows))
    lines.append("")
    lines.append("**平台期识别**：")
    if plats:
        plat_txt = format_window_index_ranges(windows, plats[:6])
        lines.append(
            f"在 **{plat_txt}** 等区间，相邻窗口间梯度较小且窗口内均值起伏有限，移动均值线形成 **平台期**"
            "（投入水平在对应时段内相对稳定）。"
        )
    else:
        lines.append("未检出足够长的「斜率持续近零且水平起伏小」区段；全课以缓变或波动为主。")
    lines.append("")
    lines.append("**突然跃迁与异常点**：")
    if jumps:
        lines.append(jump_cluster_text(jumps))
    else:
        lines.append("未检出显著的梯度尖峰，移动均值以渐进变化为主。")
    lines.append("")

    narrow_ranges, wide_ranges, thr_lo, thr_hi = extreme_bw_ranges(bws)
    lines.append("## 二、变异性与稳定性特征")
    lines.append("")
    lines.append(
        f"- **极窄带宽（高稳定性）**：{format_window_index_ranges(windows, narrow_ranges)}。"
        " 带宽极窄表示窗口内群体得分高度一致，三线贴近，系统输出可预测性强。"
    )
    lines.append(
        f"- **极宽带宽（高变异性）**：{format_window_index_ranges(windows, wide_ranges)}。"
        " 带宽拉宽表示群体分化或探索性增强，系统处于较不稳定阶段。"
    )
    lines.append(f"- **带宽变化趋势**：{bandwidth_three_segment_trend_text(windows, bws)}")
    lines.append("")

    strong_r, weak_r = attractor_segments(windows, avgs, bws, slopes, sorted_bw)
    lines.append("## 三、吸引子结构")
    lines.append("")
    mn = STRONG_ATTRACTOR_MIN_SPAN_MINUTES
    lines.append(
        "**强吸引子**：移动均值在某一水平持续平稳，且带宽极低（三线紧贴或近似重合），"
        f"且上述形态须在时间上**连续维持至少 {mn} 分钟**（按滑动窗口并段后的起止分钟计），"
        "排除转瞬即逝的偶发窄带；系统高度稳定、难以被扰动离开。"
        "**弱吸引子**：均值在同一水平附近小幅波动，带宽较窄但仍可见分离，稳定但保留一定弹性。"
    )
    lines.append("")
    lines.append(f"- **强吸引子时段**：{format_window_index_ranges(windows, strong_r)}。")
    lines.append(f"- **弱吸引子时段**：{format_window_index_ranges(windows, weak_r)}。")
    lines.append("")

    phases = find_phase_transitions(windows, avgs, bws)
    lines.append("## 四、临界点与相变定位")
    lines.append("")
    lines.append(
        "在 **低变异性（窄带）→ 高变异性鼓包 → 再回归窄带且移动均值水平与前一窄带显著不同** 的严格序列下，"
        "将中间鼓包段 **B** 的 **起点分钟** 与 **终点分钟** 标为相变 **临界点**。"
    )
    lines.append("")
    if phases:
        for idx, ph in enumerate(phases[:3], start=1):
            a0, a1 = ph.a_span
            b0, b1 = ph.b_span
            c0, c1 = ph.c_span
            lines.append(f"### 序列 {idx}")
            lines.append("")
            a_lvl, c_lvl = phase_mean_level_phrase(ph.mean_a, ph.mean_c)
            lines.append(
                f"- **旧稳定（低变异性，段 A）**：约 **{a0}-{a1} 分钟**，移动均值处于 **{a_lvl}**，"
                f"平均带宽约 **{score_str(ph.mean_bw_a)}**。"
            )
            lines.append(
                f"- **过渡期（高变异性鼓包，段 B）**：约 **{b0}-{b1} 分钟**，平均带宽约 **{score_str(ph.mean_bw_b)}**。"
            )
            lines.append(
                f"- **新稳定（低变异性新水平，段 C）**：约 **{c0}-{c1} 分钟**，移动均值处于 **{c_lvl}**，"
                f"平均带宽约 **{score_str(ph.mean_bw_c)}**（与段 A 的认知投入水平不同）。"
            )
            higher = ph.mean_c > ph.mean_a
            lines.append(
                f"- **相变结论**：系统由段 A 的 **{'较高' if not higher else '较低'}** 稳定水平 **{'抬升' if higher else '下沉'}** 至段 C 的 **{'较高' if higher else '较低'}** 稳定水平（正文不列具体分值，见图）。"
                f" {'这是一个向上相变，说明系统在高变异探索后形成了更高水平的稳定组织，学生认知有升华趋势。' if higher else '这是一个向下相变，说明学生在经历了一段探索后未实现认知升华，系统回落到更低水平。'}"
                f" **临界点**：**{b0} 分钟**-**{b1} 分钟**。"
            )
            lines.append("")
    else:
        lines.append(
            "本课数据上 **未检出** 同时满足「A 窄—B 宽鼓包—C 再窄且均值相对 A 显著变化」的完整三相序列；"
            "可能为渐变、多段扰动或阈值与课型不匹配。可将第二节极宽带宽与第一节跃迁点作 **人工相变候选** 对照课堂实录复核。"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("配图见同目录 `sliding_scores.png`。")
    return "\n".join(lines)


def write_report_for_task(repo_root: Path, task_id: str) -> bool:
    task_dir = repo_root / "tasks" / task_id
    json_path = task_dir / "analysis" / "sliding_scores.json"
    if not json_path.is_file():
        print(f"Skip {task_id}: missing {json_path}")
        return False
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    end_sec = max_lesson_end_seconds(task_dir)
    md = build_markdown(task_id, payload, end_sec)
    out_dir = task_dir / "sliding"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sliding_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    wins = payload.get("windows_3min") or []
    if wins:
        ins_path = out_dir / "sliding_insights.json"
        ins_path.write_text(
            json.dumps(build_sliding_insights_document(wins), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {ins_path}")
    return True


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    tasks_dir = repo_root / "tasks"
    if not tasks_dir.is_dir():
        print("tasks/ not found")
        sys.exit(1)
    arg = sys.argv[1] if len(sys.argv) >= 2 else None
    ids = resolve_task_ids(tasks_dir, arg)
    if not ids:
        if arg:
            print(f"No sliding_scores.json for task {arg}")
        else:
            print("No tasks with analysis/sliding_scores.json")
        sys.exit(1)
    ok = 0
    for tid in ids:
        if write_report_for_task(repo_root, tid):
            ok += 1
    print(f"Done: {ok}/{len(ids)} report(s).")


if __name__ == "__main__":
    main()

