# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive coding engagement analysis — 通过大模型 API 分析编程协作过程中的认知参与度。Python 3.13+。

## Package Manager

This project uses **uv**. Do not use pip directly.

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# Run Example
uv run src/f_role.py task_id
```

## Tasks Directory Structure

`tasks/` 存放输入视频和处理产物。每个任务一个子目录，目录名即任务 ID。

任务 ID 格式: `YYYYMMDD.HHmmss.N` — 日期 + 时间 + 自增序号（同一秒内多个任务时递增）。例如 `20260304.140500.0`。

## Processing Pipeline

`src/` 下的脚本按字母前缀排序，表示处理顺序。每个脚本接收任务 ID 作为命令行参数（未提供时自动选择最新任务）。

本项目聚焦于 f/g/h 等大模型 API 相关的处理步骤。

### f_role.py — 角色识别与标点修正

读取 `tasks/{id}/merge/output.json`，调用 GPT 判断每个语音片段的说话人角色（TEACHER/STUDENT）并补全标点。按 BATCH_SIZE=50 分批，相邻批重叠 OVERLAP=2 条以保持上下文连贯。多线程并发调用。

```bash
uv run src/f_role.py [task_id]
```

输入: `tasks/{id}/merge/output.json`
输出: `tasks/{id}/role/full.json`, `tasks/{id}/role/light.json`, `tasks/{id}/role/log.txt`

### g_group.py — 话题分组

读取 `tasks/{id}/role/full.json`，调用 GPT 按话题/知识点切换边界将片段分组。按 BATCH_SIZE=150 分批，重叠 OVERLAP=5 条，通过重叠区判断相邻批次边界是否属于同一话题，生成全局连续 group 编号。多线程并发调用。

```bash
uv run src/g_group.py [task_id]
```

输入: `tasks/{id}/role/full.json`
输出: `tasks/{id}/group/full.json`, `tasks/{id}/group/full.txt`, `tasks/{id}/group/log.txt`

### h_class.py — 认知参与分类

读取 `tasks/{id}/group/full.json` 和 `questionnaires/{qid}/student.json`、`questionnaires/{qid}/teacher.json` 两份问卷决策树，对每个 Group 分别调用 GPT 回答学生问卷和教师问卷的所有问题（yes/no），然后代码端走各自的决策树（root → JUMP/RESULT）得出认知参与类型。每个 Group 附带前后各 CONTEXT_SIZE=10 句上下文。多线程并发调用。

```bash
uv run src/h_class.py [task_id] [questionnaire_id]
```

输入: `tasks/{id}/group/full.json`, `questionnaires/{qid}/student.json`, `questionnaires/{qid}/teacher.json`
输出: `tasks/{id}/class/full.json`, `tasks/{id}/class/log.txt`

### 调试参数

三个脚本都有 DEBUG 限制参数，设为 -1 时处理全部数据：

- `f_role.py`: `DEBUG_BATCH_LIMIT`
- `g_group.py`: `DEBUG_BATCH_LIMIT`
- `h_class.py`: `DEBUG_GROUP_LIMIT`

## Language

所有输出日志用英文，代码用英文。

交流时和代码注释用中文。
