import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.gpt import call_gpt

BATCH_SIZE = 150
OVERLAP = 5
DEBUG_BATCH_LIMIT = -1  # -1 to process all batches

SYSTEM_PROMPT = "你是一位资深的小学科学课堂教学分析专家，擅长基于课堂话语结构（IRF：Initiation-Response-Feedback）对课堂对话进行分组。"

USER_PROMPT_TEMPLATE = """以下是一段课堂录音的转写片段（共{count}个片段），每个片段有一个id、说话人角色和文本内容。

请你直接按 IRF（教师发问/引导 → 学生回应 → 教师反馈）结构分组**。

定义：
- **I（Initiation，教师引导/发问）**：教师提出问题、点名提问、发出需要学生回应的指令/提示（例如“……吗？”“谁来回答？”“请说一说/想一想”）。
- **R（Response，学生回应）**：一个或多个学生对 I 的回答/回应（可为单句或多句；也可能出现多个学生接力回答或齐声回答）。
- **F（Feedback，教师反馈）**：教师对学生回应的评价、确认、纠正、澄清追问、总结/板书、或基于回应的进一步说明（如“对/不对”“很好”“注意…因为…”等）。注意：如果教师的话语本质上是**开启下一轮新问题**，则应作为下一组的 I，而不是前一组的 F。

分组规则（按优先级）：
1. **每一轮 IRF 作为一组**：从教师 I 开始，包含对应的学生 R，直到教师 F 结束，构成一个 group。
2. **新 I 必须开新组**：当教师开始提出一个新的问题/新指令（新的 I）时，即使前一轮的 F 很短或省略，也要开始新的 group。
3. **多学生回应同组**：同一轮 I 下，多个学生的连续回应（接力/齐声）都归入同一组，直到出现教师反馈 F 或下一轮 I。
4. **缺失情况处理**：
   - **只有 I 没有 R**：教师提出问题后无人回应，教师紧接着改问/提示/换问（新的 I）时，则前一个 I 自成一组（I-only），新的 I 开新组。
   - **有 R 但没有明显 F**：学生回应后教师直接进入下一个问题（新的 I），则上一组在 R 处结束（IR），新 I 开新组。
5. **教师讲解/说明的归属**：教师对学生回应后的解释、纠正、补充说明属于 F，归在该组内；若教师连续讲解且没有触发新的 I，则视作同一组的反馈延伸。
6. **crowd 片段**：crowd 类型不单独成组，归入其相邻的 IRF 组（通常跟随后面的语音片段归组即可）。
7. **组号**：组号从 0 开始，必须连续递增，不跳号。


片段列表：
{segments_text}

请你按上述 IRF 结构识别边界并分组，然后以如下格式输出JSON。每个片段都必须有对应的输出。
```json
[
  {{"id": 0, "group": 0}},
  {{"id": 1, "group": 0}},
  ...
]
```
"""


def build_batches(segments):
    """每 BATCH_SIZE 个分一批，相邻批之间重叠 OVERLAP 个片段作为上下文。"""
    batches = []
    step = BATCH_SIZE - OVERLAP
    for i in range(0, len(segments), step):
        batch = segments[i : i + BATCH_SIZE]
        batches.append(batch)
        if i + BATCH_SIZE >= len(segments):
            break
    return batches


def build_prompt(batch):
    """把一批片段组装成 prompt 文本。"""
    lines = []
    for idx, seg in enumerate(batch):
        start_sec = int(math.floor(seg.get("start", 0)))
        end_sec = int(math.ceil(seg.get("end", 0)))
        start_mm_ss = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
        end_mm_ss = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
        if seg.get("type") == "crowd":
            lines.append(f"[id: {idx}, time: {start_mm_ss}~{end_mm_ss}] (crowd noise)")
        else:
            role = seg.get("role", "UNKNOWN")
            content = seg.get("content", seg.get("text", ""))
            lines.append(f"[id: {idx}, time: {start_mm_ss}~{end_mm_ss}, role: {role}] {content}")
    segments_text = "\n".join(lines)
    return USER_PROMPT_TEMPLATE.format(count=len(batch), segments_text=segments_text)


def parse_json_response(text):
    """从 GPT 响应中提取 JSON 数组。"""
    m = re.search(r"```json\s*(\[.*?])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
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
    return batch_index, results, log


def merge_batch_results(batches, batch_results):
    """按批次顺序拼接各批的 local group，生成全局连续 group。

    策略：利用重叠区判断相邻批次边界处是否属于同一话题。
    - 重叠区最后一个 segment 在当前批的 local group 记为 anchor_group
    - 非重叠区第一个 segment 的 local group 记为 first_group
    - 如果 anchor_group == first_group，说明边界处没有话题切换，应与前一批尾 group 合并
    - 否则说明有切换，global_group 递增
    """
    all_results = {}  # segment_id -> global_group
    global_group = 0

    for batch_index in range(len(batches)):
        if batch_index not in batch_results:
            continue

        batch = batches[batch_index]
        results = batch_results[batch_index]
        # local id (batch 内 0-based index) -> local group
        local_groups = {item["id"]: item["group"] for item in results}

        skip = OVERLAP if batch_index > 0 else 0

        # 判断是否需要与前一批尾 group 合并
        should_merge = False
        if batch_index > 0 and skip > 0:
            anchor_local = local_groups.get(skip - 1)  # 重叠区最后一个
            first_local = local_groups.get(skip)  # 非重叠区第一个
            if anchor_local is not None and first_local is not None:
                should_merge = (anchor_local == first_local)

        # 如果不合并，说明边界处有话题切换，global_group 需要递增
        if batch_index > 0 and not should_merge:
            global_group += 1

        # 遍历非重叠区，local group 变化时 global_group 递增
        prev_local = None
        for idx in range(skip, len(batch)):
            local_g = local_groups.get(idx)
            if local_g is None:
                continue
            if prev_local is not None and local_g != prev_local:
                global_group += 1
            prev_local = local_g
            all_results[batch[idx]["id"]] = global_group

        # 为下一批准备：global_group 保持当前值（不递增，留给下一批判断）

    return all_results


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
    input_path = Path("tasks") / task_id / "role" / "full.json"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    batches = build_batches(segments)

    if DEBUG_BATCH_LIMIT != -1:
        batches = batches[:DEBUG_BATCH_LIMIT]
        print(f"[DEBUG] Limited to first {DEBUG_BATCH_LIMIT} batches")

    print(f"Total {len(segments)} segments, split into {len(batches)} batches")

    # 并发处理所有 batch
    batch_results = {}
    batch_logs = {}

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                _, results, log = future.result()
                batch_results[batch_index] = results
                batch_logs[batch_index] = log
            except Exception as e:
                print(f"[Batch {batch_index + 1}] Failed: {e}")

    # 拼接各批结果，生成全局连续 group
    all_results = merge_batch_results(batches, batch_results)

    # 写回 segments
    for seg in segments:
        if seg["id"] in all_results:
            seg["group"] = all_results[seg["id"]]

    # 保存输出
    output_dir = Path("tasks") / task_id / "group"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "full.json"
    full_path.write_text(
        json.dumps({"task_id": task_id, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON output saved to {full_path}")

    # 导出 txt 版本
    group_count = max((s.get("group", -1) for s in segments), default=-1) + 1
    txt_lines = []
    for g in range(group_count):
        txt_lines.append(f"Group {g}")
        for s in segments:
            if s.get("group") != g:
                continue
            start_sec = int(math.floor(s.get("start", 0)))
            end_sec = int(math.ceil(s.get("end", 0)))
            start_mm_ss = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
            end_mm_ss = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
            if s.get("type") == "crowd":
                txt_lines.append(f"[{s['id']}] [{start_mm_ss}~{end_mm_ss}] Crowd")
            else:
                role = s.get("role", "UNKNOWN").capitalize()
                content = s.get("content", s.get("text", ""))
                txt_lines.append(f"[{s['id']}] [{start_mm_ss}~{end_mm_ss}] {role}: {content}")
        txt_lines.append("")

    txt_path = output_dir / "full.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"TXT output saved to {txt_path}")

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

    # 统计
    print(f"\nTotal groups: {group_count}")
    for g in range(group_count):
        count = sum(1 for s in segments if s.get("group") == g)
        print(f"  Group {g}: {count} segments")


if __name__ == "__main__":
    main()
