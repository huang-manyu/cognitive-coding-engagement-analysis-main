import json
import math
import re
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

from models.gpt import call_gpt

CONTEXT_SIZE = 10  # 前后各取多少句作为上下文
DEBUG_GROUP_LIMIT = -1 # -1 to process all groups
JSON_PARSE_RETRIES = 3  # GPT 返回非合法 JSON 时的最大重试次数

SYSTEM_PROMPT = "你是一个课堂教学分析专家，擅长根据课堂录音转写文本分析学生和老师的认知参与行为。"

USER_PROMPT_TEMPLATE = """以下是一段课堂录音的转写片段。其中【目标片段】是你需要分析的部分，前后的【上文】和【下文】仅供你理解上下文语境。

{context_text}

【整段归类原则】目标片段可能含**多名学生或多轮学生话轮**，但决策树每个问题只回答一次 yes/no。请**纵观全段**，按**本段师生互动的认知主导与教学焦点**作答（教师主问题链指向什么、学生主要承担的认知任务是什么）。若题干写「该单句」而段内有多句学生话，以**能代表本段认知层次、且与教师当前主问题/探究主线最契合**的话轮为准，**勿**用一句孤立的低阶应答覆盖整段。

现在请你根据【目标片段】的内容，逐一回答以下问题。每个问题请：
1. 先结合目标片段的具体内容进行分析，说明你的判断依据
2. 再给出最终答案（yes 或 no）

问题列表：
{questions_text}

请你先思考分析理解整段的内容含义，回答的最后最后以如下JSON格式输出每个问题的答案，每个问题都必须回答。
analysis 字段内若需使用双引号请写成 \\"，不要使用 markdown 代码块嵌套。
```json
[
  {{"id": 1, "analysis": "...", "answer": "yes"}},
  {{"id": 2, "analysis": "...", "answer": "no"}},
  ...
]
```
"""


def load_questionnaires(questionnaire_id=None):
    """从 questionnaires/{id}/ 加载学生和教师两份问卷，未指定 id 时使用最新。"""
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
    student = json.loads((qdir / "student.json").read_text(encoding="utf-8"))
    teacher = json.loads((qdir / "teacher.json").read_text(encoding="utf-8"))
    return {"student": student, "teacher": teacher}


def format_segment(seg):
    """格式化单个 segment 为可读文本。"""
    start_sec = int(math.floor(seg.get("start", 0)))
    end_sec = int(math.ceil(seg.get("end", 0)))
    start_mm_ss = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
    end_mm_ss = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
    if seg.get("type") == "crowd":
        return f"[id:{seg['id']}, {start_mm_ss}~{end_mm_ss}] (crowd noise)"
    role = seg.get("role", "UNKNOWN")
    content = seg.get("content", seg.get("text", ""))
    return f"[id:{seg['id']}, {start_mm_ss}~{end_mm_ss}, {role}] {content}"


def build_prompt(group_segs, before_ctx, after_ctx, questions):
    """拼接完整 prompt。"""
    lines = []

    if before_ctx:
        lines.append("【上文（仅供参考）】")
        for seg in before_ctx:
            lines.append(format_segment(seg))
        lines.append("")

    lines.append("【目标片段（请分析这部分）】")
    for seg in group_segs:
        lines.append(format_segment(seg))
    lines.append("")

    if after_ctx:
        lines.append("【下文（仅供参考）】")
        for seg in after_ctx:
            lines.append(format_segment(seg))
        lines.append("")

    context_text = "\n".join(lines)

    q_lines = []
    for q in questions:
        q_lines.append(f"Q{q['id']}: {q['question']}")
    questions_text = "\n".join(q_lines)

    return USER_PROMPT_TEMPLATE.format(
        context_text=context_text, questions_text=questions_text
    )


def _fenced_json_body(text):
    """取出第一个 ```json ... ``` 代码块内部文本；若无代码块则返回原文。"""
    m = re.search(r"```(?:json)?\s*", text, re.IGNORECASE)
    if not m:
        return text
    rest = text[m.end() :]
    end = rest.find("```")
    if end != -1:
        return rest[:end]
    return rest


def _extract_top_level_json_array(s):
    """从 s 中定位最外层 [...]（忽略字符串内的括号），避免正则误截断 analysis 里的 ]。"""
    start = s.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def parse_json_response(text):
    """从 GPT 响应中提取 JSON 数组。"""
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    candidates = []
    fenced = _fenced_json_body(text)
    if fenced != text:
        candidates.append(fenced.strip())
    candidates.append(text)

    errors = []
    for raw in candidates:
        blob = _extract_top_level_json_array(raw)
        if not blob:
            continue
        try:
            data = json.loads(blob)
            if isinstance(data, list):
                return data
            errors.append("Top-level JSON is not an array")
        except json.JSONDecodeError as e:
            errors.append(str(e))

    preview = text[:400].replace("\n", " ")
    raise ValueError(
        f"Failed to parse JSON array ({'; '.join(errors) if errors else 'no [ ... ] span found'}). "
        f"Preview: {preview!r}"
    )


def walk_decision_tree(questions, answers_map):
    """根据问卷的 root/yes/no 决策树和 GPT 的回答，走出最终 RESULT。

    从 root=true 的问题开始，根据 answer 选择 yes/no 分支：
    - JUMP: 跳转到指定问题继续
    - RESULT: 返回最终结果
    """
    q_by_id = {q["id"]: q for q in questions}

    # 找到 root 问题
    root_q = next((q for q in questions if q.get("root")), None)
    if not root_q:
        return "未找到root问题", []

    current = root_q
    trace = []
    visited = set()

    while current:
        qid = current["id"]
        if qid in visited:
            trace.append({"id": qid, "question": current["question"], "note": "循环终止"})
            return "决策树循环", trace
        visited.add(qid)

        answer = answers_map.get(qid, "no").lower().strip()
        branch = current.get("yes") if answer == "yes" else current.get("no")

        trace.append({
            "id": qid,
            "question": current["question"],
            "answer": answer,
            "branch": branch,
        })

        if not branch:
            return "分支缺失", trace

        if branch["type"] == "RESULT":
            return branch["value"], trace

        # JUMP
        next_id = branch["value"]
        current = q_by_id.get(next_id)
        if not current:
            return f"跳转目标Q{next_id}不存在", trace

    return "未知", trace


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


_JSON_RETRY_SUFFIX = """

【系统解析失败后的补充要求】上一条回复无法被解析为合法 JSON。请只输出一个 JSON 数组，不要任何其它说明文字、不要用 markdown 代码块包裹。
格式严格为：[{"id": <整数>, "analysis": "<字符串>", "answer": "yes" 或 "no"}, ...]
字符串内双引号必须转义为 \\"。"""


def process_single_questionnaire(group_id, group_segs, before_ctx, after_ctx, questions, label):
    """对单个 group 跑一份问卷：构建 prompt、调用 GPT、解析结果、走决策树。"""
    base_prompt = build_prompt(group_segs, before_ctx, after_ctx, questions)
    print(f"[Group {group_id}/{label}] Calling GPT ({len(group_segs)} segments, context: {len(before_ctx)}+{len(after_ctx)})...")
    response = ""
    last_err = None
    for attempt in range(JSON_PARSE_RETRIES):
        prompt = base_prompt if attempt == 0 else base_prompt + _JSON_RETRY_SUFFIX
        response = call_gpt(prompt, system_prompt=SYSTEM_PROMPT)
        try:
            answers = parse_json_response(response)
            break
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            last_err = e
            print(f"[Group {group_id}/{label}] JSON parse failed (attempt {attempt + 1}/{JSON_PARSE_RETRIES}): {e}")
    else:
        raise last_err

    print(f"[Group {group_id}/{label}] Done, got {len(answers)} answers")

    answers_map = {a["id"]: a["answer"] for a in answers}
    result, trace = walk_decision_tree(questions, answers_map)
    print(f"[Group {group_id}/{label}] Result: {result}")

    log = {"system": SYSTEM_PROMPT, "user": prompt, "assistant": response}
    return {
        "result": result,
        "answers": answers,
        "trace": trace,
    }, log


def process_group(group_id, group_segs, all_segments, questionnaires):
    """处理单个 Group：分别跑 student 和 teacher 两份问卷。"""
    first_id = group_segs[0]["id"]
    last_id = group_segs[-1]["id"]
    before_ctx = [s for s in all_segments if first_id - CONTEXT_SIZE <= s["id"] < first_id]
    after_ctx = [s for s in all_segments if last_id < s["id"] <= last_id + CONTEXT_SIZE]

    results = {}
    logs = {}
    with ThreadPoolExecutor(max_workers=len(questionnaires)) as executor:
        futures = {
            executor.submit(
                process_single_questionnaire,
                group_id, group_segs, before_ctx, after_ctx, questions, label,
            ): label
            for label, questions in questionnaires.items()
        }
        for future in as_completed(futures):
            label = futures[future]
            res, log = future.result()
            results[label] = res
            logs[label] = log

    return {
        "group": group_id,
        "student": results["student"],
        "teacher": results["teacher"],
    }, logs


def main():
    task_id, questionnaire_id = resolve_args()
    input_path = Path("tasks") / task_id / "group" / "full.json"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    questionnaires = load_questionnaires(questionnaire_id)

    # 按 group 分组
    groups = {}
    for seg in segments:
        g = seg.get("group")
        if g is not None:
            groups.setdefault(g, []).append(seg)

    group_ids = sorted(groups.keys())

    if DEBUG_GROUP_LIMIT != -1:
        group_ids = group_ids[:DEBUG_GROUP_LIMIT]
        print(f"[DEBUG] Limited to first {DEBUG_GROUP_LIMIT} groups")

    print(f"Total {len(segments)} segments, {len(group_ids)} groups to process")

    # 多线程并行处理 group
    all_results = []
    all_logs = {}

    with ThreadPoolExecutor(max_workers=len(group_ids)) as executor:
        futures = {
            executor.submit(process_group, gid, groups[gid], segments, questionnaires): gid
            for gid in group_ids
        }
        for future in as_completed(futures):
            gid = futures[future]
            try:
                result, log = future.result()
                all_results.append(result)
                all_logs[gid] = log
            except Exception as e:
                print(f"[Group {gid}] Failed: {e}")

    # 按 group id 排序
    all_results.sort(key=lambda r: r["group"])

    # 保存输出
    output_dir = Path("tasks") / task_id / "class"
    output_dir.mkdir(parents=True, exist_ok=True)

    # full.json — 完整结果
    full_path = output_dir / "full.json"
    full_path.write_text(
        json.dumps({"task_id": task_id, "results": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON output saved to {full_path}")

    # 保存问答日志
    log_lines = []
    for gid in group_ids:
        if gid not in all_logs:
            continue
        logs = all_logs[gid]
        for label in ("student", "teacher"):
            log = logs[label]
            log_lines.append(f"{'=' * 60}")
            log_lines.append(f"Group {gid} / {label}")
            log_lines.append(f"{'=' * 60}")
            log_lines.append(f"\n[System]\n{log['system']}")
            log_lines.append(f"\n[User]\n{log['user']}")
            log_lines.append(f"\n[Assistant]\n{log['assistant']}")
            log_lines.append("")
    log_path = output_dir / "log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Log output saved to {log_path}")

    # 统计
    print(f"\nResults summary:")
    for label in ("student", "teacher"):
        print(f"  [{label}]")
        result_counts = {}
        for r in all_results:
            val = r[label]["result"]
            result_counts[val] = result_counts.get(val, 0) + 1
        for val, count in sorted(result_counts.items()):
            print(f"    {val}: {count} groups")


if __name__ == "__main__":
    main()
