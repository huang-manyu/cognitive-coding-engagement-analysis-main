import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd


def resolve_args():
    """从命令行参数获取 task_id 和 questionnaire_id，未提供时自动选择最新。"""
    task_id = sys.argv[1] if len(sys.argv) >= 2 else None
    questionnaire_id = sys.argv[2] if len(sys.argv) >= 3 else None

    if task_id is None:
        tasks_dir = Path("tasks")
        if not tasks_dir.exists():
            print("Tasks directory not found")
            sys.exit(1)
        task_dirs = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
        if not task_dirs:
            print("No tasks found")
            sys.exit(1)
        task_id = task_dirs[-1]
        print(f"No task_id provided, using latest: {task_id}")

    return task_id, questionnaire_id


def load_questionnaire_labels(questionnaire_id=None):
    """从问卷决策树中提取所有 RESULT value 作为固定类别。"""
    base = Path("questionnaires")
    if not base.exists():
        print("Questionnaires directory not found")
        sys.exit(1)
    if questionnaire_id is None:
        dirs = sorted(d.name for d in base.iterdir() if d.is_dir())
        if not dirs:
            print("No questionnaires found")
            sys.exit(1)
        questionnaire_id = dirs[-1]
        print(f"No questionnaire_id provided, using latest: {questionnaire_id}")

    qdir = base / questionnaire_id
    labels = {}
    for role in ("student", "teacher"):
        questions = json.loads((qdir / f"{role}.json").read_text(encoding="utf-8"))
        results = set()
        for q in questions:
            for branch_key in ("yes", "no"):
                branch = q.get(branch_key)
                if branch and branch.get("type") == "RESULT":
                    results.add(branch["value"])
        labels[role] = sorted(results)
    return labels


# 状态空间图：与 questionnaires 中 RESULT 维度一致；Y 轴自下而上为从低到高
STUDENT_STATE_SPACE_AXIS_ORDER = [
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

# 分析输出（热力图、状态空间 Y 轴）不包含以下两类教师 RESULT
TEACHER_RESULTS_EXCLUDED = frozenset({"质疑", "进行明确的推理论证"})

TEACHER_STATE_SPACE_AXIS_ORDER = [
    "维持秩序或与课堂无关的内容",
    "讲授",
    "引导方向",
    "表达或邀请想法",
    "联系",
    "补充想法",
    "邀请学生补充想法",
    "邀请学生推理论证",
    "想法上的协调和同意",
    "反思对话或活动",
]

# 历史 class 导出中的异名 → 当前问卷 RESULT
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


def normalize_teacher_result_for_plot(raw: str) -> str:
    if isinstance(raw, str) and raw.strip() == "反思":
        return "反思对话或活动"
    return TEACHER_RESULT_ALIASES.get(raw, raw)


def teacher_result_for_analysis(raw: str) -> str | None:
    """归一化后若在排除集合内则返回 None（不参与热力图与状态空间）。"""
    n = normalize_teacher_result_for_plot(raw)
    if n in TEACHER_RESULTS_EXCLUDED:
        return None
    return n


def build_state_space_student_labels(questionnaire_student_labels, results):
    """X：固定 9 类顺序，再追加问卷其余标签，再追加数据中独有标签。"""
    ordered, seen = [], set()
    for lab in STUDENT_STATE_SPACE_AXIS_ORDER:
        ordered.append(lab)
        seen.add(lab)
    for lab in questionnaire_student_labels:
        if lab not in seen:
            ordered.append(lab)
            seen.add(lab)
    for r in results:
        s = r["student"]["result"]
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


def build_state_space_teacher_labels(questionnaire_teacher_labels, results):
    """Y：按 TEACHER_STATE_SPACE_AXIS_ORDER 从低到高，再追加问卷/数据其余标签（先归一化，排除 TEACHER_RESULTS_EXCLUDED）。"""
    ordered, seen = [], set()
    for lab in TEACHER_STATE_SPACE_AXIS_ORDER:
        if lab in TEACHER_RESULTS_EXCLUDED:
            continue
        ordered.append(lab)
        seen.add(lab)
    for lab in questionnaire_teacher_labels:
        c = normalize_teacher_result_for_plot(lab)
        if c in TEACHER_RESULTS_EXCLUDED:
            continue
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    for r in results:
        c = teacher_result_for_analysis(r["teacher"]["result"])
        if c is None:
            continue
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


# --- 认知投入：一级 / 二级指标与权重（用于按分钟的滑动窗口得分）---
COGNITIVE_PRIMARY_ORDER = ["被动", "主动", "建构", "交互"]

PRIMARY_WEIGHTS = {
    "被动": 0.058,
    "主动": 0.248,
    "建构": 0.386,
    "交互": 0.308,
}

# 二级指标 → 所属一级
SECONDARY_PRIMARY = {
    "接受": "被动",
    "记忆": "主动",
    "应用": "主动",
    "提问": "建构",
    "阐述": "建构",
    "创造": "建构",
    "支持": "交互",
    "反对": "交互",
    "讨论": "交互",
}

# 二级指标在其一级内的权重（与用户提供值一致）
SECONDARY_WEIGHTS = {
    "接受": 1.0,
    "记忆": 0.259,
    "应用": 0.741,
    "提问": 0.175,
    "阐述": 0.262,
    "创造": 0.563,
    "支持": 0.199,
    "反对": 0.238,
    "讨论": 0.563,
}

SECONDARIES_BY_PRIMARY = {
    "被动": ["接受"],
    "主动": ["记忆", "应用"],
    "建构": ["提问", "阐述", "创造"],
    "交互": ["支持", "反对", "讨论"],
}

SLIDING_WINDOW_MINUTES = 3


def load_group_time_bounds(task_id: str):
    """从 group/full.json 得到每个 group 的起止时间（秒）及全局最大 end。"""
    group_path = Path("tasks") / task_id / "group" / "full.json"
    if not group_path.exists():
        return None, 0.0
    data = json.loads(group_path.read_text(encoding="utf-8"))
    bounds = defaultdict(lambda: [float("inf"), float("-inf")])
    for seg in data.get("segments", []):
        g = seg["group"]
        bounds[g][0] = min(bounds[g][0], float(seg["start"]))
        bounds[g][1] = max(bounds[g][1], float(seg["end"]))
    if not bounds:
        return {}, 0.0
    max_end = max(b[1] for b in bounds.values())
    return {k: (v[0], v[1]) for k, v in bounds.items()}, max_end


def minute_overlap_weights(t0: float, t1: float, num_minutes: int) -> list[tuple[int, float]]:
    """
    For a group spanning [t0, t1) in seconds, return (minute_index, weight) pairs.
    Weight = (overlap length in that minute) / (t1 - t0), so weights sum to 1 when the
    interval lies inside [0, num_minutes * 60). Used instead of attributing the whole
    group to int(t0 // 60) only (which zeros out later minutes for long groups).
    """
    t0, t1 = float(t0), float(t1)
    if t1 <= t0:
        m = int(t0 // 60)
        if 0 <= m < num_minutes:
            return [(m, 1.0)]
        return []
    dur = t1 - t0
    out: list[tuple[int, float]] = []
    for m in range(num_minutes):
        lo, hi = m * 60.0, (m + 1) * 60.0
        ov = max(0.0, min(t1, hi) - max(t0, lo))
        if ov > 0:
            out.append((m, ov / dur))
    return out


def primary_score_from_minute_counts(counts: Counter, primary: str) -> float:
    """一级得分 = Σ(二级频数/该一级下二级频数之和 × 二级权重)。"""
    secs = SECONDARIES_BY_PRIMARY[primary]
    denom = sum(counts[s] for s in secs)
    if denom <= 0:
        return 0.0
    return sum((counts[s] / denom) * SECONDARY_WEIGHTS[s] for s in secs)


def total_engagement_score_for_minute(counts: Counter) -> float:
    """总分 = Σ(一级得分 × 一级权重)。"""
    return sum(
        primary_score_from_minute_counts(counts, p) * PRIMARY_WEIGHTS[p]
        for p in COGNITIVE_PRIMARY_ORDER
    )


def main():
    task_id, questionnaire_id = resolve_args()
    input_path = Path("tasks") / task_id / "class" / "full.json"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    # 以下热力图、按分钟折线图、状态空间网格的分类均来自此处（h_class 输出）。
    # 折线图：仅用每条 r["student"]["result"]；状态空间：学生 + 教师 r["teacher"]["result"]（教师侧做归一化）。
    results = data.get("results", [])

    # 从问卷读取所有可能的类别
    labels = load_questionnaire_labels(questionnaire_id)
    student_labels = labels["student"]
    teacher_labels = labels["teacher"]
    teacher_labels_plot = [t for t in teacher_labels if t not in TEACHER_RESULTS_EXCLUDED]

    # 统计 (student_result, teacher_result)；教师侧仅保留未排除标签（归一化后与问卷键一致）
    counts = Counter()
    for r in results:
        s = r["student"]["result"]
        t_raw = r["teacher"]["result"]
        t = teacher_result_for_analysis(t_raw)
        if t is None:
            continue
        counts[(s, t)] += 1

    # 构建矩阵 DataFrame，横轴学生，纵轴老师（不含排除的教师类别）
    matrix = []
    for t_label in teacher_labels_plot:
        row = [counts.get((s_label, t_label), 0) for s_label in student_labels]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=teacher_labels_plot, columns=student_labels)

    # 绘制热力图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    cell_size = 1.2
    fig, ax = plt.subplots(
        figsize=(len(student_labels) * cell_size + 3, len(teacher_labels_plot) * cell_size + 2)
    )
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
                annot_kws={"size": 9}, square=True)
    ax.set_xlabel("学生认知投入", fontsize=10, labelpad=12)
    ax.set_ylabel("教师言语策略", fontsize=10, labelpad=12)
    ax.set_title("师生互动数量矩阵", fontsize=12, pad=20)

    # 用 transform 偏移 tick labels，使其远离方块
    dx, dy = 0, -8 / 72  # 8pt 向下
    x_offset = mtransforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + x_offset)

    dx, dy = -8 / 72, 0  # 8pt 向左
    y_offset = mtransforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
    for label in ax.get_yticklabels():
        label.set_transform(label.get_transform() + y_offset)

    # 保存
    output_dir = Path("tasks") / task_id / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")

    # 打印统计
    print(f"\nMatrix ({len(teacher_labels_plot)} teacher x {len(student_labels)} student, excluded teacher labels omitted):")
    print(df.to_string())

    # --- 按分钟的认知投入得分 + 3 分钟滑动窗口折线图 ---
    # 分类：仅 class results 中的学生二级指标；时间：group 里该 group 的起止（与状态空间共用 group_bounds）。
    group_bounds, max_end_sec = load_group_time_bounds(task_id)
    minute_detail = []
    windows_out = []

    if group_bounds is None or not results:
        print("Skip sliding window chart: missing group/full.json or empty class results.")
    else:
        num_minutes = max(1, int(math.ceil(max_end_sec / 60.0)))
        # 每分钟：按 group 时间轴 [t0,t1] 与每分钟区间重叠比例，把权重分摊到各分钟（每条 class result 权重和为 1）
        minute_counters = [Counter() for _ in range(num_minutes)]
        skipped_unknown = 0
        skipped_no_group = 0
        for r in results:
            gid = r["group"]
            lab = r["student"]["result"]
            if gid not in group_bounds:
                skipped_no_group += 1
                continue
            t0, t1 = group_bounds[gid]
            if lab not in SECONDARY_WEIGHTS:
                skipped_unknown += 1
                continue
            for m, w in minute_overlap_weights(t0, t1, num_minutes):
                minute_counters[m][lab] += w

        if skipped_unknown:
            print(f"Note: {skipped_unknown} groups skipped (student result not in 9 secondaries).")
        if skipped_no_group:
            print(f"Note: {skipped_no_group} groups skipped (group id not in group/full.json).")
        print(
            "Per-minute student labels use time-weighted allocation from each group's [start,end] "
            "(overlap / group duration)."
        )

        minute_scores = []
        for m in range(num_minutes):
            cnt = minute_counters[m]
            p_scores = {p: primary_score_from_minute_counts(cnt, p) for p in COGNITIVE_PRIMARY_ORDER}
            total = total_engagement_score_for_minute(cnt)
            minute_scores.append(total)
            freq_9 = {s: round(float(cnt[s]), 6) for s in sorted(SECONDARY_WEIGHTS.keys())}
            minute_detail.append({
                "minute_1based": m + 1,
                "secondary_frequencies": freq_9,
                "primary_scores": {k: round(v, 6) for k, v in p_scores.items()},
                "total_score": round(total, 6),
            })

        # 滑动窗口：第 w 个窗口 = 第 w+1, w+2, w+3 分钟（1-based），步长 1
        if num_minutes >= SLIDING_WINDOW_MINUTES:
            for w in range(num_minutes - SLIDING_WINDOW_MINUTES + 1):
                chunk = minute_scores[w : w + SLIDING_WINDOW_MINUTES]
                m_start, m_end = w + 1, w + SLIDING_WINDOW_MINUTES
                windows_out.append({
                    "minutes_1based": list(range(m_start, m_end + 1)),
                    "scores": [round(s, 6) for s in chunk],
                    "min": round(min(chunk), 6),
                    "avg": round(sum(chunk) / len(chunk), 6),
                    "max": round(max(chunk), 6),
                })
        elif num_minutes > 0:
            chunk = minute_scores[:]
            windows_out.append({
                "minutes_1based": list(range(1, num_minutes + 1)),
                "scores": [round(s, 6) for s in chunk],
                "min": round(min(chunk), 6),
                "avg": round(sum(chunk) / len(chunk), 6),
                "max": round(max(chunk), 6),
            })

        payload = {
            "num_minutes": num_minutes,
            "per_minute": minute_detail,
            "windows_3min": windows_out,
        }
        windows_path = output_dir / "sliding_scores.json"
        windows_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Sliding scores saved to {windows_path}")

        if windows_out:
            x_labels = [f"{w['minutes_1based'][0]}-{w['minutes_1based'][-1]}" for w in windows_out]
            x = range(len(windows_out))
            mins_s = [w["min"] for w in windows_out]
            avgs_s = [w["avg"] for w in windows_out]
            maxs_s = [w["max"] for w in windows_out]

            fig2, ax2 = plt.subplots(figsize=(max(len(windows_out) * 0.5, 8), 5))
            ax2.plot(x, mins_s, marker="o", markersize=4, label="极小值", linewidth=1.5)
            ax2.plot(x, avgs_s, marker="s", markersize=4, label="移动均值", linewidth=1.5)
            ax2.plot(x, maxs_s, marker="^", markersize=4, label="极大值", linewidth=1.5)
            ax2.set_xticks(list(x))
            ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax2.set_xlabel("分钟", fontsize=10)
            ax2.set_ylabel("学生认知投入度", fontsize=10)
            ax2.set_title("学生认知投入度移动极值图", fontsize=12)
            ax2.legend(fontsize=9)
            all_y_vals = [*mins_s, *avgs_s, *maxs_s]
            y_data_max = max(all_y_vals) if all_y_vals else 0.0
            y_top = max(0.4, math.ceil(y_data_max / 0.05) * 0.05)
            ax2.set_ylim(0, y_top)
            ax2.yaxis.set_major_locator(MultipleLocator(0.05))
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()

            line_path = output_dir / "sliding_scores.png"
            fig2.savefig(line_path, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"Sliding scores chart saved to {line_path}")
        else:
            print("No sliding windows to plot (insufficient minutes).")

    # --- 状态空间网格图（轴顺序与 questionnaires 一致；Y 自下而上从低到高）---
    # 分类：同一份 class results 的学生 + 教师；散点时长：同上 group_bounds（不再次读 group/full.json）。
    if group_bounds is None:
        print("Skip state space grid: missing group/full.json.")
    elif not results:
        print("Skip state space grid: empty class results.")
    else:
        ss_student_labels = build_state_space_student_labels(student_labels, results)
        ss_teacher_labels = build_state_space_teacher_labels(teacher_labels, results)

        s_label_to_idx = {l: i + 1 for i, l in enumerate(ss_student_labels)}
        t_label_to_idx = {l: i + 1 for i, l in enumerate(ss_teacher_labels)}

        trajectory = []
        durations = []
        for r in results:
            gid = r["group"]
            s_res = r["student"]["result"]
            t_res = teacher_result_for_analysis(r["teacher"]["result"])
            if t_res is None:
                continue
            if gid not in group_bounds:
                print(f"Warning: skip group {gid} for state space (group id not in group/full.json)")
                continue
            if s_res not in s_label_to_idx or t_res not in t_label_to_idx:
                print(f"Warning: skip group {gid} for state space (unknown label: {s_res!r} / {t_res!r})")
                continue
            trajectory.append((s_label_to_idx[s_res], t_label_to_idx[t_res]))
            t0, t1 = group_bounds[gid]
            durations.append(max(t1 - t0, 1.0))

        if not trajectory:
            print("No trajectory points for state space grid (skipped).")
        else:
            dur_min, dur_max = min(durations), max(durations)
            if dur_max > dur_min:
                sizes = [160 + (d - dur_min) / (dur_max - dur_min) * 1440 for d in durations]
            else:
                sizes = [800] * len(durations)

            x_coords = [p[0] for p in trajectory]
            y_coords = [p[1] for p in trajectory]
            n_s = len(ss_student_labels)
            n_t = len(ss_teacher_labels)

            fig3, ax3 = plt.subplots(figsize=(n_s * cell_size + 3, n_t * cell_size + 2))

            ax3.scatter(x_coords[1:], y_coords[1:], s=sizes[1:], c="skyblue", edgecolors="none", alpha=0.5, zorder=3)
            ax3.scatter(x_coords[0], y_coords[0], s=sizes[0], facecolors="none", edgecolors="#4a6670", linewidths=1, zorder=4)

            radii_pt = [math.sqrt(s) / 2 for s in sizes]
            for i in range(len(trajectory) - 1):
                ax3.annotate("", xy=trajectory[i + 1], xytext=trajectory[i],
                             arrowprops=dict(arrowstyle="->", color="black", lw=0.6,
                                             shrinkA=radii_pt[i] + 1,
                                             shrinkB=radii_pt[i + 1] + 1,
                                             connectionstyle="arc3,rad=0"))

            ax3.set_xticks(range(1, n_s + 1))
            ax3.set_yticks(range(1, n_t + 1))
            ax3.set_xticklabels(ss_student_labels, rotation=35, ha="right", fontsize=11)
            ax3.set_yticklabels(ss_teacher_labels, fontsize=11)

            ax3.set_xticks([i + 0.5 for i in range(n_s + 1)], minor=True)
            ax3.set_yticks([i + 0.5 for i in range(n_t + 1)], minor=True)
            ax3.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.15)
            ax3.grid(which="major", visible=False)

            ax3.set_xlim(0.5, n_s + 0.5)
            ax3.set_ylim(0.5, n_t + 0.5)

            ax3.set_xlabel("学生认知投入度", fontsize=13, labelpad=14)
            ax3.set_ylabel("教师言语策略", fontsize=13, labelpad=14)
            ax3.set_title("师生互动状态空间网格图", fontsize=15, pad=22)

            ssg_path = output_dir / "state_space_grid.png"
            fig3.savefig(ssg_path, dpi=300, bbox_inches="tight")
            plt.close(fig3)
            print(f"State space grid saved to {ssg_path}")


if __name__ == "__main__":
    main()
