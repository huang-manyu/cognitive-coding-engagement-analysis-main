import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.gpt import call_gpt

BATCH_SIZE = 50
OVERLAP = 2
DEBUG_BATCH_LIMIT = -1  # -1 to process all batches

SYSTEM_PROMPT = "你是一个语言分析专家，了解中学课堂和青少年的说话性格，擅长分析课堂录音转写文本，判断说话人角色并修正文本标点。"

USER_PROMPT_TEMPLATE = """以下是一段课堂录音的转写片段（共{count}个片段），每个片段有一个id和对应的文本内容及说话人标识。

请按顺序完成以下任务（输出时每个 id 都必须保留一条记录，与输入条数一致）：

1. **纠错**：依据上下文改正错别字；若为科学课等学科场景，可顺带统一、规范相关术语。
2. **噪音处理**：若某片段可判定为明显不属于核心对话的噪音（如开场白、问候语、识别错误的背景音，例如“小哥”“第一老师您好”），则在该条的 content 中填空字符串 ""，role 仍标为 TEACHER 或 STUDENT 中更合理的一个；不要省略该 id 的整条输出，以便与后续流程对齐。
3. **角色**：根据上下文语义，判断每个片段的说话人是老师(TEACHER)还是学生(STUDENT)。
4. **标点与格式**：给每个片段的文本内容加上合适的标点符号，写入 content 字段；整段使用简体中文。

片段列表：
{segments_text}

请你先整体概括这段课堂讲了什么，然后分析和理解具体对话内容，仔细思考，最后再以如下格式输出JSON。
```json
[
  {{"id": 0, "role": "TEACHER", "content": "..."}},
  {{"id": 1, "role": "STUDENT", "content": "..."}},
  ...
]
```
"""


def build_batches(segments):
    """过滤 crowd 类型，每 BATCH_SIZE 个分一批，相邻批之间重叠 OVERLAP 个片段作为上下文。"""
    filtered = [s for s in segments if s.get("type") != "crowd"]
    batches = []
    step = BATCH_SIZE - OVERLAP
    for i in range(0, len(filtered), step):
        batch = filtered[i : i + BATCH_SIZE]
        batches.append(batch)
        if i + BATCH_SIZE >= len(filtered):
            break
    return batches


def build_prompt(batch):
    """把一批片段组装成 prompt 文本。"""
    lines = []
    for idx, seg in enumerate(batch):
        text = seg.get("text", "")
        start_sec = int(math.floor(seg.get("start", 0)))
        end_sec = int(math.ceil(seg.get("end", 0)))
        start_mm_ss = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
        end_mm_ss = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
        lines.append(f"[id: {idx}, time: {start_mm_ss}~{end_mm_ss}] {text}")
    segments_text = "\n".join(lines)
    return USER_PROMPT_TEMPLATE.format(count=len(batch), segments_text=segments_text)


def parse_json_response(text):
    """从 GPT 响应中提取 JSON 数组。"""
    # 尝试匹配 ```json ... ``` 代码块
    m = re.search(r"```json\s*(\[.*?])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 尝试匹配裸 JSON 数组
    m = re.search(r"\[.*]", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Failed to parse JSON from response:\n{text[:200]}")


def process_batch(batch_index, batch):
    """处理单个 batch：构建 prompt、调用 GPT、返回解析结果和完整问答上下文。"""
    prompt = build_prompt(batch)
    print(f"[Batch {batch_index + 1}] Calling GPT ({len(batch)} segments)...")
    response = call_gpt(prompt, system_prompt=SYSTEM_PROMPT)
    results = parse_json_response(response)
    print(f"[Batch {batch_index + 1}] Done, got {len(results)} results")
    log = {"system": SYSTEM_PROMPT, "user": prompt, "assistant": response}
    return batch_index, batch, results, log


def resolve_task_id():
    """从命令行参数获取 task_id，未提供时自动选择最新的任务。"""
    if len(sys.argv) >= 2:
        return sys.argv[1]
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
    return task_id


def main():
    task_id = resolve_task_id()
    input_path = Path("tasks") / task_id / "merge" / "output.json"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    batches = build_batches(segments)

    if DEBUG_BATCH_LIMIT != -1:
        batches = batches[:DEBUG_BATCH_LIMIT]
        print(f"[DEBUG] Limited to first {DEBUG_BATCH_LIMIT} batches")

    print(f"Total {len(segments)} segments, split into {len(batches)} batches after filtering crowd")

    # 收集实际处理的 segment 集合（用于 debug 模式只输出处理过的）
    processed_keys = set()

    # 用 (start, end) 作为 key 建立原始 segment 的索引
    seg_lookup = {}
    for seg in segments:
        key = (seg["start"], seg["end"])
        seg_lookup[key] = seg

    # 多线程并发调用 GPT
    batch_logs = {}
    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                _, batch, results, log = future.result()
                batch_logs[batch_index] = log
            except Exception as e:
                print(f"[Batch {batch_index + 1}] Failed: {e}")
                continue

            # 用 id 索引回 batch 中的原始片段，写入 role 和 content
            results_by_id = {item["id"]: item for item in results}
            # 重叠区域：非首批的前 OVERLAP 条跳过，以前一批结果为准
            skip = OVERLAP if batch_index > 0 else 0
            for idx in range(skip, len(batch)):
                if idx not in results_by_id:
                    continue
                seg = batch[idx]
                key = (seg["start"], seg["end"])
                if key in seg_lookup:
                    seg_lookup[key]["role"] = results_by_id[idx]["role"]
                    seg_lookup[key]["content"] = results_by_id[idx]["content"]
                    processed_keys.add(key)

    # 输出：debug 模式只保留处理过的 speech segments + 全部 crowd segments
    if DEBUG_BATCH_LIMIT != -1:
        out_segments = [
            s for s in segments
            if s.get("type") == "crowd" or (s["start"], s["end"]) in processed_keys
        ]
    else:
        out_segments = segments

    # 保存输出
    output_dir = Path("tasks") / task_id / "role"
    output_dir.mkdir(parents=True, exist_ok=True)

    # full.json — 完整字段
    full_segments = [{"id": i, **s} for i, s in enumerate(out_segments)]
    full_path = output_dir / "full.json"
    full_path.write_text(
        json.dumps({"task_id": task_id, "segments": full_segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull output saved to {full_path}")

    # light.json — 精简字段
    light_segments = []
    for i, s in enumerate(out_segments):
        if s.get("type") == "crowd":
            light_segments.append({"id": i, "type": "crowd", "start": s["start"], "end": s["end"]})
        else:
            item = {
                "id": i,
                "type": s.get("type"),
                "start": s["start"],
                "end": s["end"],
                "speaker": s.get("speaker"),
                "is_crowd": s.get("is_crowd"),
                "role": s.get("role"),
                "content": s.get("content"),
            }
            if s.get("overlap_count", 0) > 0:
                item["overlap_count"] = s["overlap_count"]
                item["overlapping_speakers"] = s.get("overlapping_speakers", [])
            light_segments.append(item)
    light_path = output_dir / "light.json"
    light_path.write_text(
        json.dumps({"task_id": task_id, "segments": light_segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Light output saved to {light_path}")

    # 保存完整问答日志，按 batch 顺序合并
    log_lines = []
    for i in range(len(batches)):
        if i not in batch_logs:
            continue
        log = batch_logs[i]
        log_lines.append(f"{'=' * 60}")
        log_lines.append(f"Batch {i + 1}")
        log_lines.append(f"{'=' * 60}")
        log_lines.append(f"\n[System]\n{log['system']}")
        log_lines.append(f"\n[User]\n{log['user']}")
        log_lines.append(f"\n[Assistant]\n{log['assistant']}")
        log_lines.append("")
    log_path = output_dir / "log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Log output saved to {log_path}")


if __name__ == "__main__":
    main()
