from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler

# 树状图横轴：task_id → 课题名称（未列出的任务仍显示原 ID）
TASK_DISPLAY_LABELS: Dict[str, str] = {
    "20260304.140500.0": "《食物中的营养》",
    "20260324.111100.0": "《摆的快慢》",
    "20260324.111300.0": "《呼吸与空气》",
    "20260324.111400.0": "《灵活巧妙的剪刀》",
    "20260324.111500.0": "《水到哪里去了》",
    "20260324.111600.0": "《地球、月球和太阳》",
    "20260324.120100.0": "《撬杠的学问》",
    "20260324.130100.0": "《八颗行星》",
    "20260324.130200.0": "《影子的秘密》",
    "20260324.130300.0": "《气体的热胀冷缩》",
    "20260324.130400.0": "《保护土壤资源》",
    "20260324.130500.0": "《河流和湖泊》",
    "20260324.130600.0": "《土壤的组成》",
    "20260324.130700.0": "《怎样才省力》",
    "20260324.130800.0": "《哪个传热快》",
    "20260324.130900.0": "《让资源再生》",
    "20260324.131000.0": "《相貌各异的我们》",
    "20260403.190000.0": "《食物在身体里的旅行》",
    "20260403.190100.0": "《用浮的材料造船》",
    "20260403.190200.0": "《我们是怎样听到声音的》",
    "20260403.190300.0": "《制作我的小乐器》",
    "20260404.190400.0": "《昼夜的交替》",
    "20260404.190500.0": "《用沉的材料造船》",
}


def task_label_for_dendrogram(task_id: str) -> str:
    return TASK_DISPLAY_LABELS.get(task_id, task_id)


def estimate_num_minutes(task_dir: Path, ins: Dict[str, Any]) -> int:
    """估计单课总分钟数（优先从 sliding_scores.json，其次从各区间最大 end_minute 推断）。"""
    json_path = task_dir / "analysis" / "sliding_scores.json"
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            nm = int(payload.get("num_minutes", 0))
            if nm > 0:
                return nm
        except Exception:
            # 日志保持英文
            print(f"Warn: failed to read num_minutes from {json_path}")

    max_m = 0
    for key in ("plateaus", "strong_attractor_spans", "weak_attractor_spans"):
        for sp in ins.get(key) or []:
            try:
                m = int(sp.get("end_minute", 0))
            except Exception:
                m = 0
            if m > max_m:
                max_m = m
    for ph in ins.get("phase_transitions") or []:
        for part in ("a_span", "b_span", "c_span"):
            sp = ph.get(part) or {}
            try:
                m = int(sp.get("end_minute", 0))
            except Exception:
                m = 0
            if m > max_m:
                max_m = m
    return max_m


def _total_span_minutes(spans: List[Dict[str, Any]]) -> int:
    """计算若干分钟区间的总分钟数。"""
    total = 0
    for sp in spans:
        try:
            a = int(sp.get("start_minute", 0))
            b = int(sp.get("end_minute", a))
        except Exception:
            a, b = 0, 0
        if a and b and b >= a:
            total += b - a + 1
    return total


def extract_features_from_insights(task_dir: Path, ins: Dict[str, Any]) -> List[float] | None:
    """从单课 sliding_insights.json 中提取动力学特征向量。"""
    num_minutes = estimate_num_minutes(task_dir, ins)

    jumps = ins.get("jumps") or []
    plateaus = ins.get("plateaus") or []
    strong = ins.get("strong_attractor_spans") or []
    weak = ins.get("weak_attractor_spans") or []
    phases = ins.get("phase_transitions") or []

    # 波动 / 跃迁相关
    num_jumps_total = len(jumps)
    num_jumps_up = sum(1 for j in jumps if j.get("direction") == "up")
    num_jumps_down = sum(1 for j in jumps if j.get("direction") == "down")
    jump_up_ratio = num_jumps_up / max(1, num_jumps_total)

    if jumps:
        mins = []
        for j in jumps:
            try:
                mins.append(int(j.get("minute", 0)))
            except Exception:
                continue
        if mins:
            first_jump_minute = min(mins)
            last_jump_minute = max(mins)
        else:
            first_jump_minute = 0
            last_jump_minute = 0
    else:
        first_jump_minute = 0
        last_jump_minute = 0

    # 吸引子 / 平台相关
    strong_minutes = _total_span_minutes(strong)
    weak_minutes = _total_span_minutes(weak)
    plateau_minutes = _total_span_minutes(plateaus)

    attr_coverage_ratio = (
        (strong_minutes + weak_minutes) / num_minutes if num_minutes > 0 else 0.0
    )

    strong_blocks = len(strong)
    weak_blocks = len(weak)

    first_strong_minute, last_strong_minute = 0, 0
    if strong:
        starts: List[int] = []
        ends: List[int] = []
        for sp in strong:
            try:
                starts.append(int(sp.get("start_minute", 0)))
                ends.append(int(sp.get("end_minute", 0)))
            except Exception:
                continue
        if starts and ends:
            first_strong_minute = min(starts)
            last_strong_minute = max(ends)

    # 相变相关
    num_phases = len(phases)
    num_phase_up = 0
    num_phase_down = 0
    for ph in phases:
        d = ph.get("direction")
        if d == "up":
            num_phase_up += 1
        elif d == "down":
            num_phase_down += 1
    phase_up_ratio = num_phase_up / max(1, num_phases)

    if phases:
        first_dir = phases[0].get("direction")
        if first_dir == "up":
            first_phase_type = 1.0
        elif first_dir == "down":
            first_phase_type = -1.0
        else:
            first_phase_type = 0.0
    else:
        first_phase_type = 0.0

    phase_span_total_minutes = 0
    critical_points: List[int] = []
    for ph in phases:
        b = ph.get("b_span") or {}
        try:
            b0 = int(b.get("start_minute", 0))
            b1 = int(b.get("end_minute", b0))
        except Exception:
            b0, b1 = 0, 0
        if b0 and b1 and b1 >= b0:
            phase_span_total_minutes += b1 - b0 + 1
        for c in ph.get("critical_minutes") or []:
            try:
                critical_points.append(int(c))
            except Exception:
                continue

    if critical_points:
        phase_critical_minute_mean = float(sum(critical_points)) / len(critical_points)
    else:
        phase_critical_minute_mean = 0.0

    feats: List[float] = [
        float(num_minutes),
        float(num_jumps_total),
        float(num_jumps_up),
        float(num_jumps_down),
        float(jump_up_ratio),
        float(first_jump_minute),
        float(last_jump_minute),
        float(strong_minutes),
        float(weak_minutes),
        float(plateau_minutes),
        float(attr_coverage_ratio),
        float(strong_blocks),
        float(weak_blocks),
        float(first_strong_minute),
        float(last_strong_minute),
        float(num_phases),
        float(num_phase_up),
        float(num_phase_down),
        float(phase_up_ratio),
        float(first_phase_type),
        float(phase_span_total_minutes),
        float(phase_critical_minute_mean),
    ]
    return feats


def load_all_features(tasks_dir: Path) -> Tuple[List[str], List[List[float]]]:
    """遍历 tasks 目录，加载所有含 sliding_insights.json 的任务特征。"""
    task_ids: List[str] = []
    features: List[List[float]] = []

    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        ins_path = d / "sliding" / "sliding_insights.json"
        if not ins_path.is_file():
            continue
        try:
            ins = json.loads(ins_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip {d.name}: failed to read {ins_path} ({e})")
            continue

        vec = extract_features_from_insights(d, ins)
        if vec is None:
            continue

        task_ids.append(d.name)
        features.append(vec)

    return task_ids, features


def plot_dendrogram(Z: np.ndarray, task_ids: List[str], out_path: Path) -> None:
    """绘制层次聚类树状图。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    # 宽度随样本数自适应，避免标签挤在一起
    width = max(8.0, len(task_ids) * 0.25)
    plt.figure(figsize=(width, 6.0))
    leaf_labels = [task_label_for_dendrogram(tid) for tid in task_ids]
    dendrogram(Z, labels=leaf_labels, leaf_rotation=90)
    ax = plt.gca()
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def write_labels_json(out_path: Path, task_ids: List[str], labels: np.ndarray) -> None:
    """写出每个 task 的聚类标签。"""
    mapping: Dict[str, int] = {tid: int(lbl) for tid, lbl in zip(task_ids, labels)}
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_cluster_report(
    out_path: Path,
    task_ids: List[str],
    labels: np.ndarray,
    X: np.ndarray,
) -> None:
    """根据聚类结果生成中文动力学画像报告。"""
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(labels):
        clusters.setdefault(int(cid), []).append(idx)

    n_total = len(task_ids)

    lines: List[str] = []
    lines.append("# 课堂认知动力学聚类报告")
    lines.append("")
    lines.append(
        "本报告基于各课堂在 `sliding_insights.json` 中的波动、吸引子与相变特征，"
        "提取突变次数与方向、吸引子时长与分布、相变次数与方向及其临界点位置等动力学指标，"
        "使用层次聚类（Ward + 欧氏距离）对所有课堂进行分群，并给出每一类的整体画像。"
    )
    lines.append("")

    for cid in sorted(clusters.keys()):
        idxs = clusters[cid]
        n_c = len(idxs)
        frac = n_c / n_total if n_total > 0 else 0.0
        X_c = X[idxs, :]

        means = X_c.mean(axis=0)
        (
            mean_num_minutes,
            mean_num_jumps_total,
            mean_num_jumps_up,
            mean_num_jumps_down,
            mean_jump_up_ratio,
            mean_first_jump,
            mean_last_jump,
            mean_strong_minutes,
            mean_weak_minutes,
            mean_plateau_minutes,
            mean_attr_cov,
            mean_strong_blocks,
            mean_weak_blocks,
            mean_first_strong,
            mean_last_strong,
            mean_num_phases,
            mean_num_phase_up,
            mean_num_phase_down,
            mean_phase_up_ratio,
            mean_first_phase_type,
            mean_phase_span_total,
            mean_phase_crit_mean,
        ) = means

        lines.append(f"## 类别 {cid}")
        lines.append("")
        lines.append(f"- **样本数量**：本类包含 {n_c} 节课，占全部样本约 {frac:.1%}。")

        # 波动特征
        lines.append(
            f"- **波动特征**：本类课堂平均检测到约 {mean_num_jumps_total:.1f} 次显著梯度跃迁，"
            f"其中上升跃迁占比约 {mean_jump_up_ratio:.1%}，下降跃迁约 {mean_num_jumps_down:.1f} 次。"
        )
        lines.append(
            f"  典型的第一次跃迁出现在约第 {mean_first_jump:.1f} 分钟附近，"
            f"最后一次跃迁大致在第 {mean_last_jump:.1f} 分钟左右。"
        )

        # 吸引子结构
        lines.append(
            f"- **吸引子与平台结构**：强吸引子累计约 {mean_strong_minutes:.1f} 分钟，"
            f"弱吸引子约 {mean_weak_minutes:.1f} 分钟，平台期累计约 {mean_plateau_minutes:.1f} 分钟。"
        )
        lines.append(
            f"  吸引子覆盖比例平均约为 {mean_attr_cov:.1%}，"
            f"强吸引子段数约 {mean_strong_blocks:.1f} 段、弱吸引子约 {mean_weak_blocks:.1f} 段。"
        )
        lines.append(
            f"  强吸引子往往从约第 {mean_first_strong:.1f} 分钟开始显著出现，"
            f"延续至约第 {mean_last_strong:.1f} 分钟附近。"
        )

        # 相变模式
        lines.append(
            f"- **相变模式**：本类课堂平均检出 {mean_num_phases:.1f} 条严格 A–B–C 相变序列，"
            f"其中向上相变次数约 {mean_num_phase_up:.1f} 次，向下相变约 {mean_num_phase_down:.1f} 次，"
            f"向上相变占比约 {mean_phase_up_ratio:.1%}。"
        )
        lines.append(
            f"  所有相变鼓包段（B 段）累计时长约 {mean_phase_span_total:.1f} 分钟，"
            f"临界点整体集中在约第 {mean_phase_crit_mean:.1f} 分钟附近。"
        )

        if mean_first_phase_type > 0.2:
            first_phase_desc = "大多以向上相变为起始"
        elif mean_first_phase_type < -0.2:
            first_phase_desc = "大多以向下相变为起始"
        else:
            first_phase_desc = "起始相变方向较为混合或多数课堂未形成完整相变"

        lines.append(
            f"- **综合动力学画像**：整体上，本类课堂 {first_phase_desc}，"
            "在波动扰动与吸引子停留之间交替，呈现出相对稳定的课堂动力学类型。"
        )

        # 若干代表性课堂编号
        sample_ids = [task_ids[i] for i in idxs[: min(6, len(idxs))]]
        if sample_ids:
            lines.append(f"- **示例课堂编号**：`{', '.join(sample_ids)}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    """命令行入口：运行聚类、输出树状图与报告。"""
    repo_root = Path(__file__).resolve().parent.parent
    tasks_dir = repo_root / "tasks"
    if not tasks_dir.is_dir():
        print("tasks/ not found")
        sys.exit(1)

    task_ids, features = load_all_features(tasks_dir)
    if not task_ids:
        print("No tasks with sliding/sliding_insights.json")
        sys.exit(1)

    X = np.asarray(features, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 层次聚类
    Z = linkage(X_scaled, method="ward", metric="euclidean")

    out_dir = tasks_dir / "cluster"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 树状图
    dendro_path = out_dir / "dendrogram.png"
    plot_dendrogram(Z, task_ids, dendro_path)

    # 默认聚成 4 类，如需调整可在命令行传参
    k = 4
    if len(sys.argv) >= 2:
        try:
            k = int(sys.argv[1])
        except Exception:
            print(f"Warn: invalid cluster count '{sys.argv[1]}', fallback to k={k}")

    labels = fcluster(Z, t=k, criterion="maxclust")

    # 标签与报告
    labels_path = out_dir / "labels.json"
    write_labels_json(labels_path, task_ids, labels)

    report_path = out_dir / "cluster_report.md"
    write_cluster_report(report_path, task_ids, labels, X)

    print(f"Done: {len(task_ids)} task(s), {k} cluster(s).")


if __name__ == "__main__":
    main()

