"""
教师侧认知策略混淆矩阵：读取 `tasks/教师正确率计算.csv`（可为 .xls 伪扩展名），
输出频数 CSV、行归一化（0–1，两位小数）CSV 与热力图 PNG 到独立目录。
矩阵不含「质疑」「进行明确的推理论证」行/列；含该标签的样本行不参与统计。

风格与路径约定参考 `m_cluster_sliding_metrics.py`（repo_root、tasks 子路径）。

用法：
  uv run src/m_teacher_confusion_matrix.py
  uv run src/m_teacher_confusion_matrix.py --input tasks/教师正确率计算.csv --out-dir tasks/teacher_confusion_matrix

Logs in English.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# questionnaires/*/teacher.json 中 RESULT 的稳定顺序；热力图/CSV 排除下列两类
TEACHER_MATRIX_EXCLUDE: frozenset[str] = frozenset({"质疑", "进行明确的推理论证"})

_TEACHER_LABELS_FULL: list[str] = [
    "维持秩序或与课堂无关的内容",
    "讲授",
    "引导方向",
    "想法上的协调和同意",
    "补充想法",
    "质疑",
    "进行明确的推理论证",
    "联系",
    "反思对话或活动",
    "表达或邀请想法",
    "邀请学生补充想法",
    "邀请学生推理论证",
]

TEACHER_LABELS: list[str] = [x for x in _TEACHER_LABELS_FULL if x not in TEACHER_MATRIX_EXCLUDE]


def _is_zip_xlsx(path: Path) -> bool:
    try:
        return path.read_bytes()[:4] == b"PK\x03\x04"
    except OSError:
        return False


def _is_xls_ole(path: Path) -> bool:
    try:
        return path.read_bytes()[:4] == b"\xd0\xcf\x11\xe0"
    except OSError:
        return False


def read_teacher_table(path: Path, encoding: str | None, sheet: str | int) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".xlsx" or _is_zip_xlsx(path):
        print(f"Note: reading as .xlsx (openpyxl): {path}")
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if suf == ".xls" or _is_xls_ole(path):
        print(f"Note: reading as .xls (xlrd): {path}")
        return pd.read_excel(path, sheet_name=sheet, engine="xlrd")

    if encoding:
        return pd.read_csv(path, encoding=encoding)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "utf-16", "utf-16-le"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err else OSError(f"Cannot read {path}")


def build_teacher_confusion(
    df: pd.DataFrame,
    col_true: str,
    col_pred: str,
    labels: list[str],
) -> pd.DataFrame:
    missing = [c for c in (col_true, col_pred) if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        print("Available:", list(df.columns))
        sys.exit(1)

    sub = df[[col_true, col_pred]].copy()
    sub[col_true] = sub[col_true].astype(str).str.strip()
    sub[col_pred] = sub[col_pred].astype(str).str.strip()
    sub = sub.dropna(subset=[col_true, col_pred])

    label_set = set(labels)
    mask = sub[col_true].isin(label_set) & sub[col_pred].isin(label_set)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Note: {dropped} rows skipped (label not in teacher label set).")
    sub = sub.loc[mask]

    cm = pd.crosstab(
        sub[col_true],
        sub[col_pred],
        rownames=["真实值(人工)"],
        colnames=["预测值(机器)"],
    )
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.index.name = "真实值(人工)"
    cm.columns.name = "预测值(机器)"
    return cm.astype(int)


def row_proportion_01(cm: pd.DataFrame) -> pd.DataFrame:
    rs = cm.sum(axis=1).replace(0, 1)
    return (cm.div(rs, axis=0)).round(2)


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_teacher_heatmap(
    data: pd.DataFrame,
    title: str,
    annot_fmt: str,
    out_path: Path,
    figsize: tuple[float, float] = (14, 11),
    dpi: int = 200,
) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        data,
        annot=True,
        fmt=annot_fmt,
        cmap="Blues",
        linewidths=0,
        square=False,
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_xlabel("预测值", fontsize=14)
    ax.set_ylabel("真实值", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_in = root / "tasks" / "教师正确率计算.csv"
    default_out = root / "tasks" / "teacher_confusion_matrix"
    p = argparse.ArgumentParser(description="Teacher strategy confusion matrix from spreadsheet.")
    p.add_argument("--input", type=Path, default=default_in, help="Path to .xls / .xlsx / CSV")
    p.add_argument("--encoding", type=str, default=None, help="CSV encoding if not Excel")
    p.add_argument(
        "--sheet",
        type=str,
        default="0",
        help='Excel sheet index or name (default "0")',
    )
    p.add_argument("--col-true", type=str, default="人工编码", help="Human / gold label column")
    p.add_argument("--col-pred", type=str, default="机器编码", help="Model / predicted column")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output folder for CSV/PNG")
    p.add_argument("--basename", type=str, default="teacher_confusion_matrix", help="Output file stem")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path: Path = args.input
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    sheet: str | int = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = read_teacher_table(path, args.encoding, sheet)
    df.columns = [str(c).strip() for c in df.columns]

    cm = build_teacher_confusion(df, args.col_true, args.col_pred, TEACHER_LABELS)
    prop = row_proportion_01(cm)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.basename
    csv_counts = out_dir / f"{base}_counts.csv"
    csv_prop = out_dir / f"{base}_row_proportion.csv"
    png_counts = out_dir / f"{base}_counts.png"
    png_prop = out_dir / f"{base}_row_proportion.png"

    cm.to_csv(csv_counts, encoding="utf-8-sig")
    prop.to_csv(csv_prop, encoding="utf-8-sig")
    plot_teacher_heatmap(cm, "教师策略混淆矩阵（频数）", "d", png_counts)
    plot_teacher_heatmap(prop, "教师言语策略混淆矩阵", ".2f", png_prop)

    print(f"Wrote {csv_counts}")
    print(f"Wrote {csv_prop}")
    print(f"Wrote {png_counts}")
    print(f"Wrote {png_prop}")


if __name__ == "__main__":
    main()
