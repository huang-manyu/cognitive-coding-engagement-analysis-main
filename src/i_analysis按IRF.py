import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
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

TEACHER_STATE_SPACE_AXIS_ORDER = [
    "维持秩序或与课堂无关的内容",
    "讲授",
    "引导方向",
    "表达或邀请想法",
    "联系",
    "补充想法",
    "邀请学生补充想法",
    "进行明确的推理论证",
    "邀请学生推理论证",
    "质疑",
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
    """Y：按 TEACHER_STATE_SPACE_AXIS_ORDER 从低到高，再追加问卷/数据其余标签（先归一化）。"""
    ordered, seen = [], set()
    for lab in TEACHER_STATE_SPACE_AXIS_ORDER:
        ordered.append(lab)
        seen.add(lab)
    for lab in questionnaire_teacher_labels:
        c = normalize_teacher_result_for_plot(lab)
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    for r in results:
        c = normalize_teacher_result_for_plot(r["teacher"]["result"])
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def main():
    task_id, questionnaire_id = resolve_args()
    input_path = Path("tasks") / task_id / "class" / "full.json"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    # 从问卷读取所有可能的类别
    labels = load_questionnaire_labels(questionnaire_id)
    student_labels = labels["student"]
    teacher_labels = labels["teacher"]

    # 统计 (student_result, teacher_result) 出现次数
    counts = Counter()
    for r in results:
        s = r["student"]["result"]
        t = r["teacher"]["result"]
        counts[(s, t)] += 1

    # 构建矩阵 DataFrame，横轴学生，纵轴老师
    matrix = []
    for t_label in teacher_labels:
        row = [counts.get((s_label, t_label), 0) for s_label in student_labels]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=teacher_labels, columns=student_labels)

    # 绘制热力图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    cell_size = 1.2
    fig, ax = plt.subplots(figsize=(len(student_labels) * cell_size + 3, len(teacher_labels) * cell_size + 2))
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
    print(f"\nMatrix ({len(teacher_labels)} teacher x {len(student_labels)} student):")
    print(df.to_string())

    # --- 滑动窗口得分折线图 ---
    # 学生类别对应的贡献分
    score_map = {
        "接受": 1,
        "记忆": 2,
        "应用": 3,
        "提问": 4,
        "阐述": 5,
        "创造": 6,
        "支持": 7,
        "反对": 8,
        "讨论": 9,
    }

    # 每个 group 的学生得分
    group_scores = []
    for r in results:
        s = r["student"]["result"]
        group_scores.append(score_map.get(s, 0))

    # 滑动窗口（窗口大小 5，步长 1）
    window = 5
    windows = []
    for i in range(len(group_scores) - window + 1):
        chunk = group_scores[i : i + window]
        windows.append({
            "groups": list(range(i, i + window)),
            "scores": chunk,
            "min": min(chunk),
            "avg": round(sum(chunk) / len(chunk), 4),
            "max": max(chunk),
        })

    # 保存 JSON
    windows_path = output_dir / "sliding_scores.json"
    windows_path.write_text(json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sliding scores saved to {windows_path}")

    # 绘制折线图（横轴为窗口序号 1、2、3…）
    x = range(len(windows))
    x_labels = [str(i + 1) for i in x]
    mins = [w["min"] for w in windows]
    avgs = [w["avg"] for w in windows]
    maxs = [w["max"] for w in windows]

    fig2, ax2 = plt.subplots(figsize=(max(len(windows) * 0.5, 8), 5))
    ax2.plot(x, mins, marker="o", markersize=4, label="最低分", linewidth=1.5)
    ax2.plot(x, avgs, marker="s", markersize=4, label="平均分", linewidth=1.5)
    ax2.plot(x, maxs, marker="^", markersize=4, label="最高分", linewidth=1.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_xlabel("滑动窗口序号", fontsize=10)
    ax2.set_ylabel("认知投入度", fontsize=10)
    ax2.set_title("学生认知投入度移动极值图", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    line_path = output_dir / "sliding_scores.png"
    fig2.savefig(line_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Sliding scores chart saved to {line_path}")

    # --- 状态空间网格图（轴顺序与 questionnaires 一致；Y 自下而上从低到高）---
    group_path = Path("tasks") / task_id / "group" / "full.json"
    if not group_path.exists():
        print(f"Group file not found (skip state space grid): {group_path}")
    else:
        group_data = json.loads(group_path.read_text(encoding="utf-8"))
        group_times = defaultdict(lambda: [float("inf"), float("-inf")])
        for seg in group_data["segments"]:
            g = seg["group"]
            group_times[g][0] = min(group_times[g][0], seg["start"])
            group_times[g][1] = max(group_times[g][1], seg["end"])

        ss_student_labels = build_state_space_student_labels(student_labels, results)
        ss_teacher_labels = build_state_space_teacher_labels(teacher_labels, results)

        s_label_to_idx = {l: i + 1 for i, l in enumerate(ss_student_labels)}
        t_label_to_idx = {l: i + 1 for i, l in enumerate(ss_teacher_labels)}

        trajectory = []
        durations = []
        for r in results:
            gid = r["group"]
            s_res = r["student"]["result"]
            t_res = normalize_teacher_result_for_plot(r["teacher"]["result"])
            if s_res not in s_label_to_idx or t_res not in t_label_to_idx:
                print(f"Warning: skip group {gid} for state space (unknown label: {s_res!r} / {t_res!r})")
                continue
            trajectory.append((s_label_to_idx[s_res], t_label_to_idx[t_res]))
            times = group_times[gid]
            durations.append(max(times[1] - times[0], 1.0))

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
            ax3.set_xticklabels(ss_student_labels, rotation=35, ha="right", fontsize=8)
            ax3.set_yticklabels(ss_teacher_labels, fontsize=8)

            ax3.set_xticks([i + 0.5 for i in range(n_s + 1)], minor=True)
            ax3.set_yticks([i + 0.5 for i in range(n_t + 1)], minor=True)
            ax3.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.15)
            ax3.grid(which="major", visible=False)

            ax3.set_xlim(0.5, n_s + 0.5)
            ax3.set_ylim(0.5, n_t + 0.5)

            ax3.set_xlabel("学生认知投入度", fontsize=10, labelpad=12)
            ax3.set_ylabel("教师言语策略", fontsize=10, labelpad=12)
            ax3.set_title("师生互动状态空间网格图", fontsize=12, pad=20)

            ssg_path = output_dir / "state_space_grid.png"
            fig3.savefig(ssg_path, dpi=300, bbox_inches="tight")
            plt.close(fig3)
            print(f"State space grid saved to {ssg_path}")


if __name__ == "__main__":
    main()
