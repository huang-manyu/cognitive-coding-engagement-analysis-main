"""
从 CSV（或误用 .csv 扩展名的 Excel 工作簿）生成学生认知投入 9 类混淆矩阵，
逻辑与 D:\\confusion_matrix_from_excel.py 一致。

默认：`tasks/我的正确率计算.csv`；当前表头为 `real labels` / `predicted labels`
（与参考 Excel 脚本的 `p_label` / `predict` 不同，可用参数覆盖）。
若文件实为 xlsx（ZIP 头 PK），自动用 read_excel；否则按 utf-8-sig / gbk / utf-16 读 CSV。
行归一化矩阵为每行除以行和，取值 [0,1]，保留两位小数（非 0–100%）。

默认输出目录：`tasks/student_confusion_matrix/`（频数/归一化 CSV 与 PNG）。

用法：
  uv run scripts/confusion_matrix_from_csv.py
  uv run scripts/confusion_matrix_from_csv.py --csv "tasks/我的正确率计算.csv" --out-dir tasks/自定义目录
  # 若需写回数据同目录：--out-dir tasks（或与 --csv 相同的父目录）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 与问卷 / 参考脚本一致的 9 类顺序（行=真实值，列=预测值）
STUDENT_LABELS = [
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_csv = root / "tasks" / "我的正确率计算.csv"
    default_out = root / "tasks" / "student_confusion_matrix"
    p = argparse.ArgumentParser(description="从 CSV 生成混淆矩阵图与 CSV")
    p.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="CSV 路径（默认：tasks/我的正确率计算.csv）",
    )
    p.add_argument("--encoding", type=str, default=None, help="强制编码；默认自动探测（仅 CSV）")
    p.add_argument(
        "--sheet",
        type=str,
        default="0",
        help='Excel 工作表名或数字索引（默认 "0"；仅当文件按 xlsx 读取时有效）',
    )
    p.add_argument(
        "--col-true",
        type=str,
        default="real labels",
        help="真实标签列名（参考脚本里常用 p_label）",
    )
    p.add_argument(
        "--col-pred",
        type=str,
        default="predicted labels",
        help="预测标签列名（参考脚本里常用 predict）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="输出目录（默认：tasks/student_confusion_matrix）",
    )
    p.add_argument(
        "--basename",
        type=str,
        default="confusion_matrix",
        help="输出文件主名（不含扩展名）",
    )
    return p.parse_args()


def _is_zip_xlsx(path: Path) -> bool:
    try:
        head = path.read_bytes()[:4]
    except OSError:
        return False
    return head.startswith(b"PK\x03\x04")


def read_table_flexible(
    path: Path,
    encoding: str | None,
    sheet: str | int,
) -> pd.DataFrame:
    """CSV 多编码尝试；实为 xlsx 时读第一个工作表（可用 --sheet 指定）。"""
    if path.suffix.lower() in (".xlsx", ".xls") or _is_zip_xlsx(path):
        print(f"Note: reading as Excel workbook (openpyxl): {path}")
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

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


def build_confusion_counts(
    df: pd.DataFrame, col_true: str, col_pred: str
) -> pd.DataFrame:
    missing = [c for c in (col_true, col_pred) if c not in df.columns]
    if missing:
        print("CSV 中未找到列:", missing)
        print("当前列名:", list(df.columns))
        sys.exit(1)

    sub = df[[col_true, col_pred]].copy()
    sub[col_true] = sub[col_true].astype(str).str.strip()
    sub[col_pred] = sub[col_pred].astype(str).str.strip()
    sub = sub.dropna(subset=[col_true, col_pred])

    mask = sub[col_true].isin(STUDENT_LABELS) & sub[col_pred].isin(STUDENT_LABELS)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Note: {dropped} rows skipped (label not in 9-class set).")
    sub = sub.loc[mask]

    cm = pd.crosstab(
        sub[col_true],
        sub[col_pred],
        rownames=["真实值"],
        colnames=["预测值"],
    )
    cm = cm.reindex(index=STUDENT_LABELS, columns=STUDENT_LABELS, fill_value=0)
    cm.index.name = "真实值"
    cm.columns.name = "预测值"
    return cm.astype(int)


def row_proportion_01(cm: pd.DataFrame) -> pd.DataFrame:
    """每行除以行和，得到 [0,1] 比例，保留两位小数。"""
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


def plot_heatmap(
    data: pd.DataFrame,
    title: str,
    annot_fmt: str,
    out_path: Path,
    figsize: tuple[float, float] = (10, 7),
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
        square=True,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_xlabel("预测值", fontsize=12)
    ax.set_ylabel("真实值", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv
    if not csv_path.is_file():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    sheet: str | int = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = read_table_flexible(csv_path, args.encoding, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    cm = build_confusion_counts(df, args.col_true, args.col_pred)
    prop = row_proportion_01(cm)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.basename
    csv_counts = out_dir / f"{base}_counts.csv"
    csv_prop = out_dir / f"{base}_row_proportion.csv"
    png_counts = out_dir / f"{base}_counts.png"
    png_prop = out_dir / f"{base}_row_proportion.png"

    cm.to_csv(csv_counts, encoding="utf-8-sig")
    prop.to_csv(csv_prop, encoding="utf-8-sig")

    plot_heatmap(cm, "混淆矩阵（频数）", "d", png_counts)
    plot_heatmap(prop, "学生认知投入度混淆矩阵", ".2f", png_prop)

    print(f"Saved: {csv_counts}")
    print(f"Saved: {csv_prop}")
    print(f"Saved: {png_counts}")
    print(f"Saved: {png_prop}")
    print("\n混淆矩阵（频数）:")
    print(cm.to_string())


if __name__ == "__main__":
    main()
