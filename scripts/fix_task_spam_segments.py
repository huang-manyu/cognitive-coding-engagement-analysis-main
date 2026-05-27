"""Remove known ad/spam segments and bogus tail timestamps; renumber ids. One-off helper."""
import json
import math
import sys
from pathlib import Path

SPAM_SUBSTRINGS = ("请不吝点赞", "优优独播", "YoYo Television")


def is_spam(seg: dict) -> bool:
    t = (seg.get("text") or "") + (seg.get("content") or "")
    if any(s in t for s in SPAM_SUBSTRINGS):
        return True
    # role/light.json 常无 text，广告段可能只有空 content + 典型时间窗
    st = float(seg.get("start") or 0)
    en = float(seg.get("end") or 0)
    if 929.9 <= st <= 930.1 and 959.0 <= en <= 961.0:
        return True
    return False


def clean_json(path: Path, renumber: bool, merge_count: bool) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    segs = [s for s in d["segments"] if not is_spam(s)]
    if renumber:
        for i, s in enumerate(segs):
            s["id"] = i
    d["segments"] = segs
    if merge_count and "segment_count" in d:
        d["segment_count"] = len(segs)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(segs)


def write_group_txt(task_dir: Path) -> None:
    data = json.loads((task_dir / "group/full.json").read_text(encoding="utf-8"))
    segs = data["segments"]
    group_count = max((s.get("group", -1) for s in segs), default=-1) + 1
    lines = []
    for g in range(group_count):
        lines.append(f"Group {g}")
        for s in segs:
            if s.get("group") != g:
                continue
            start_sec = int(math.floor(s.get("start", 0)))
            end_sec = int(math.ceil(s.get("end", 0)))
            start_mm_ss = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
            end_mm_ss = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
            if s.get("type") == "crowd":
                lines.append(f"[{s['id']}] [{start_mm_ss}~{end_mm_ss}] Crowd")
            else:
                role = s.get("role", "UNKNOWN").capitalize()
                content = s.get("content", s.get("text", ""))
                lines.append(f"[{s['id']}] [{start_mm_ss}~{end_mm_ss}] {role}: {content}")
        lines.append("")
    (task_dir / "group/full.txt").write_text("\n".join(lines), encoding="utf-8")


def rebuild_role_light_from_full(task_dir: Path) -> None:
    """role/light.json 与 full 片段数对齐（仅保留轻量字段）。"""
    fp = task_dir / "role/full.json"
    lp = task_dir / "role/light.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    raw = data["segments"]
    slimmed = []
    i = 0
    while i < len(raw):
        s = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        # full 中常见：同时间窗一条 is_crowd 的 speech + 一条 crowd，light 只保留 crowd
        if (
            s.get("type") == "speech"
            and s.get("is_crowd")
            and nxt
            and nxt.get("type") == "crowd"
            and float(s.get("start", -1)) == float(nxt.get("start", -2))
            and float(s.get("end", -1)) == float(nxt.get("end", -2))
        ):
            slimmed.append(
                {"type": "crowd", "start": nxt["start"], "end": nxt["end"]}
            )
            i += 2
            continue
        if s.get("type") == "crowd":
            slimmed.append({"type": "crowd", "start": s["start"], "end": s["end"]})
            i += 1
            continue
        item = {
            "type": s.get("type", "speech"),
            "start": s["start"],
            "end": s["end"],
            "speaker": s.get("speaker"),
            "is_crowd": s.get("is_crowd", False),
            "role": s.get("role"),
            "content": s.get("content"),
        }
        if "overlap_count" in s:
            item["overlap_count"] = s["overlap_count"]
        if "overlapping_speakers" in s:
            item["overlapping_speakers"] = s["overlapping_speakers"]
        slimmed.append(item)
        i += 1
    for j, s in enumerate(slimmed):
        s["id"] = j
    out = {"task_id": data.get("task_id", task_dir.name), "segments": slimmed}
    lp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "20260324.111500.0"
    task_dir = Path("tasks") / task_id

    p = task_dir / "merge/output.json"
    n = clean_json(p, renumber=False, merge_count=True)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["task_id"] = task_id
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merge/output.json -> {n} segments")

    for name in ("group/full.json", "role/full.json"):
        print(f"{name} -> {clean_json(task_dir / name, renumber=True, merge_count=False)} segments")

    rebuild_role_light_from_full(task_dir)
    print("role/light.json rebuilt from role/full.json")

    write_group_txt(task_dir)
    print("group/full.txt regenerated")


if __name__ == "__main__":
    main()
