# Folium

> 基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 二开的科研智能体改造项目。

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Folium 当前目标是在一个极简 AI 编程 Agent 的基础上，逐步改造成面向科研场景的智能体系统。项目会围绕科研调研、报告生成、论文写作、实验代码生成与执行，以及 Agent Harness 工程组件持续扩展。

## 当前能力

- Web 对话界面：支持新建对话、切换历史对话、流式响应和工具调用展示
- CLI 入口：支持交互式对话、单次 prompt、会话恢复和内置命令
- OpenAI 兼容模型接入：通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`FOLIUM_MODEL` 配置模型
- Agent 循环：模型可以多轮调用工具，再基于工具结果继续推理
- 工具系统：支持读文件、写文件、搜索、编辑、执行 shell 命令和子 Agent，执行前会统一校验工具参数，并对长时间无响应的工具调用做超时兜底
- 会话持久化：对话内容保存到项目内 `conversations/`
- 上下文压缩：三层渐进式压缩（截断工具输出、占位符压缩、LLM 摘要）
- Token 统计：实时显示上下文窗口占用、本轮用量、会话累计（含缓存命中率和费用）
- 本地可观测性：记录一次用户输入触发的 Agent 执行 trace、LLM 调用、工具调用和上下文压缩

## 运行方式

安装为本地可编辑包：

```bash
pip install -e .
```

在项目根目录创建 `.env`，或在 shell 中设置环境变量：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
FOLIUM_MODEL=deepseek-chat
```

启动 Web 界面：

```bash
python -m folium
```

默认访问：

```text
http://localhost:8000
```

启动 CLI：

```bash
python -m folium --cli
```

单次任务：

```bash
python -m folium --cli -p "读一下 README.md，总结当前项目能力"
```

## 配置

常用环境变量：

```text
OPENAI_API_KEY              模型 API key
OPENAI_BASE_URL             OpenAI 兼容 API 地址
FOLIUM_API_KEY              Folium 专用 API key，优先级高于 OPENAI_API_KEY
FOLIUM_BASE_URL             Folium 专用 base URL
FOLIUM_MODEL                默认模型，默认 gpt-4o
FOLIUM_PROVIDER             openai 或 litellm
FOLIUM_MAX_TOKENS           单次输出 token 上限
FOLIUM_TEMPERATURE          采样温度
FOLIUM_MAX_CONTEXT          上下文 token 上限，默认 1000000
FOLIUM_TOKEN_ESTIMATOR      无真实 usage 时的 token 估算器：deepseek 或 approx，默认 deepseek
FOLIUM_DEEPSEEK_TOKENIZER   DeepSeek 官方 tokenizer 本地路径，默认估算器会优先使用它
```

如果使用 Ollama 这类本地 OpenAI 兼容服务：

```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
FOLIUM_MODEL=qwen3:32b
```

## Web 界面

Web 入口提供：

- 新建对话
- 历史对话列表
- 切换和删除历史对话
- 流式输出
- 工具调用标签
- 上下文窗口占用（真实 token，带百分比进度条）
- 本轮用量（输入、输出、缓存、费用）
- 会话累计（总 token、费用、轮次柱状图）
- 历史统计（缓存命中、输出、总费用、缓存命中率）
- 常用命令按钮：帮助、Token 用量、压缩上下文

当前行为说明：

- 点击“新建对话”会进入一个空白对话状态
- 空白对话不会立刻持久化
- 发送第一条消息后，对话才会保存到 `conversations/`
- 切换到其他对话前，当前已有内容的对话会自动保存

## CLI 命令

```text
/help           查看帮助
/model          查看当前模型
/model <名称>   切换模型
/tokens         查看累计 token 用量和费用估算
/compact        手动压缩上下文
/diff           查看当前会话修改过的文件
/save           手动保存会话
/sessions       列出已保存会话
/traces         列出最近执行 trace
/trace <id>     查看某个 trace 摘要
/reset          清空当前对话历史
quit            退出
```

## 工具

内置工具位于 `folium/tools/`：

```text
read_file       读取文件，支持 offset/limit
write_file      创建或覆盖文件
edit_file       基于唯一字符串匹配的安全编辑，返回 diff
glob            按 glob 模式查找文件
grep            按正则搜索文件内容
bash            执行 shell 命令，包含危险命令拦截、超时终止进程和输出截断
agent           启动子 Agent 处理独立子任务
```

工具接口保持简单：

```python
from folium.tools.base import Tool

class HttpTool(Tool):
    name = "http"
    description = "请求一个 URL。"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def execute(self, url: str) -> str:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()[:5000]
```

工具调用流程：

```text
模型输出工具调用
-> 根据工具名找到 Tool
-> 使用 Tool.validate_arguments() 按 parameters 校验参数
-> 校验通过后调用 execute()
-> 校验失败时返回参数错误，不执行真实工具
```

如果连续 5 次工具参数校验失败，Agent 会停止当前任务并返回：

```text
连续 5 次工具调用失败，已停止当前任务。
```

新增工具时需要：

- 继承 `folium.tools.base.Tool`
- 声明 `name`、`description`、`parameters`
- 在 `parameters` 中使用 JSON Schema object 描述参数，并写清 `required`
- 实现 `execute`
- 注册到 `folium/tools/__init__.py` 的 `ALL_TOOLS`
- 补充 schema/参数校验测试，有副作用的工具还要覆盖主要成功和失败路径

## Skills

Folium 支持轻量级 skill。Skill 位于项目根目录的 `skills/`，每个 skill 使用一个目录和一个 `SKILL.md`：

```text
skills/
└── your-skill-name/SKILL.md
```

启动时 Agent 会扫描 `skills/*/SKILL.md`，只把 skill 的名称、描述和文件路径加入系统提示词。完整 `SKILL.md` 不会预先塞进上下文；模型在判断任务匹配某个 skill 时，会先用 `read_file` 读取对应文件，再按其中的工作流执行。

`SKILL.md` 需要包含简单 frontmatter：

```markdown
---
name: your-skill-name
description: Use when the user asks for this specialized workflow.
---

# Your Skill Name

...
```

## 上下文压缩

Folium 采用三级渐进式上下文压缩策略，自动触发时优先使用 LLM API 返回的 usage，并对新增内容使用本地估算：

| 层级 | 触发阈值 | 操作 | 成本 |
|------|---------|------|------|
| Layer 1 | 60% / 70% | 截断长 tool 输出（primary 工具 60%，secondary 工具 70%，保留首尾各 1536 字符） | 零 LLM 调用 |
| Layer 2 | 80% | 将已截断的 primary/default tool 输出替换为占位符 | 零 LLM 调用 |
| Layer 3 | 90% | 增量更新历史摘要，并按预算保护最近用户原文 | 调 LLM，失败时本地兜底 |

触发时机：
- 用户消息加入后（`after_user_message`）
- 工具结果加入后（`after_tool_results`）

Token 计算：
- 自动触发时优先使用最近一次 LLM API 返回的真实 `prompt_tokens` 和 `completion_tokens`
- 新加入的消息（用户输入或工具结果）、压缩后的 `after_tokens`、首次调用前以及手动 `/compact` 使用本地估算器
- 默认估算器为 `deepseek`；配置 `FOLIUM_DEEPSEEK_TOKENIZER` 后，会优先使用 DeepSeek 官方 tokenizer，加载失败时自动回退到 `approx`（兼容原有 `len(text) // 3`）。如需强制使用旧估算方式，可设置 `FOLIUM_TOKEN_ESTIMATOR=approx`
- 压缩水位按输入预算判断：`输入预算 = FOLIUM_MAX_CONTEXT - 20000`，默认给模型输出预留 20000 tokens 缓冲
- 压缩后不再保留 recent tail；系统会从真实用户消息中倒序保护最近原文，默认预算 20000 tokens。若最近一条用户消息本身超过预算，也会整条保留。该预算使用同一个本地 token 估算器：优先 tokenizer，失败再回退 `len(text) // 3`
- `bash`、`grep`、`glob` 等 primary 工具在 60% 后可裁剪；`read_file`、`agent` 等 secondary 工具在 70% 后可裁剪，但 80% 的占位符压缩会跳过 secondary 工具

费用计算：
- 支持缓存 token 单独计费（`prompt_cache_hit_tokens`）
- 费用公式：`(prompt - cached) × 输入价 + cached × 缓存价 + completion × 输出价`

## 本地可观测性

Folium 已经加入本地 JSONL trace 记录。一次用户输入会生成一个 trace，一次 LLM 调用、工具调用、Agent round、上下文压缩会生成对应 span 或 event。

默认保存位置：

```text
conversations/traces/
```

当前记录内容包括：

- `user_task`：一次用户输入触发的完整 Agent 执行
- `agent_round`：每轮 Agent 循环
- `llm`：模型调用、消息数量、工具数量、token、首 token 时间、输出摘要
- `tool`：工具名称、参数、结果摘要、耗时、错误状态
- `context_compression`：上下文压缩前后的 token 和消息数量
- `agent_result`：最终回复摘要、消息数量和上下文 token 估算

可观测性配置：

```text
FOLIUM_OBSERVABILITY=1
FOLIUM_TRACE_MODE=all
FOLIUM_TRACE_FULL_USER_INPUT=1
FOLIUM_TRACE_FULL_LLM_INPUT=0
FOLIUM_TRACE_FULL_LLM_OUTPUT=0
FOLIUM_TRACE_FULL_TOOL_ARGS=1
FOLIUM_TRACE_FULL_TOOL_OUTPUT=0
FOLIUM_TRACE_REDACT_SECRETS=1
FOLIUM_TRACE_MAX_PREVIEW_CHARS=1000
```

查看 trace：

```text
/traces
/trace <trace_id>
```

## 项目结构

```text
folium/
├── __main__.py              Web/CLI 入口分发
├── cli.py                   CLI REPL 和命令
├── agent.py                 Agent 主循环、工具调用和观测插桩
├── llm.py                   OpenAI 兼容 LLM 客户端和 LiteLLM 后端
├── context.py               上下文估算与压缩
├── session.py               会话保存、读取、切换和删除
├── config.py                环境变量配置
├── prompt.py                系统提示词
├── observability/           本地 trace、span、脱敏和摘要读取
├── tools/                   内置工具
└── web/
    ├── server.py            FastAPI + SSE 后端
    └── static/index.html    Web 前端
```

## 科研智能体改造方向

后续计划围绕科研工作流和 Harness 组件继续扩展：

- 科研主题调研
- 结构化报告生成
- TeX 论文写作
- Python 实验代码生成与运行
- Agentic RAG 和证据链追踪
- 沙箱执行与文件影响追踪
- Artifact 记录，如报告、TeX、代码、图表、实验日志
- 评估与反馈机制
- Langfuse、Phoenix 或 OpenTelemetry 等外部观测集成

## 测试

当前可用的 unittest：

```bash
python -m unittest tests.test_tool_validation tests.test_tool_encoding tests.test_observability tests.test_web_server tests.test_session
```

如果安装了 pytest，也可以运行：

```bash
pytest
```

## License

MIT。基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 二开，原作者何宇峰。
