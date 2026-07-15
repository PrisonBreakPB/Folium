# Folium

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Folium 面向科研任务的完整链路：从围绕研究主题检索和梳理文献、识别已有工作的条件与局限，到检查数学推导和控制理论中的关键结论，再到生成、运行和分析仿真实验，并辅助组织论文写作内容。项目也会同步建设支撑这些工作流的 Agent Harness 工程组件，包括工具调用与状态的可观测性、受控的沙箱执行、检索与记忆结合的 Agentic RAG，以及面向多步骤科研任务的上下文管理和可靠性机制。

## 当前能力

- Web 对话界面：支持新建对话、切换历史对话、流式响应、工具调用展示和 todo 状态展示
- OpenAI 兼容模型接入：通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`FOLIUM_MODEL` 配置模型
- Agent 循环：模型可以多轮调用工具，再基于工具结果继续推理；多步骤任务会通过 todo 工具维护当前进度
- 工具系统：支持读文件、写文件、本地搜索、Web 搜索、网页读取、编辑、执行 shell 命令、子 Agent 和 todo 列表，执行前会统一校验工具参数，并对长时间无响应的工具调用做超时兜底
- 本地持久化：会话、完整消息、trace 与 trace event 统一保存到 SQLite 数据库 `data/folium.db`
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
FOLIUM_BASH_BACKEND         bash 工具执行后端：local 或 docker，默认 docker；如需本地执行可显式设为 local
FOLIUM_HOST_WORKSPACE       真实项目目录，默认当前进程工作目录
FOLIUM_SANDBOX_WORKSPACE_MODE 工作区模式：host 或 copy；Web 默认 copy。copy 会在每个 Web 会话复制项目到 .folium/sandbox/sessions，文件改动不会自动回写真实项目
FOLIUM_DOCKER_IMAGE         Docker 沙箱镜像，默认 python:3.11-slim
FOLIUM_DOCKER_NETWORK       Docker 沙箱网络模式，默认 bridge；如需禁止 bash 容器联网可设为 none
FOLIUM_DOCKER_CPUS          Docker 沙箱 CPU 限制，默认 1
FOLIUM_DOCKER_MEMORY        Docker 沙箱内存限制，默认 2g
BRAVE_SEARCH_API_KEY        Brave Search API key，用于 web_search 工具
FOLIUM_MEMORY_MAINTENANCE_TURNS               Turns without memory maintenance before scheduling (default 10)
FOLIUM_MEMORY_MAINTENANCE_MAX_STEPS           Maximum maintenance model/tool rounds (default 5)
FOLIUM_MEMORY_MAINTENANCE_MAX_TOKENS          Background model output limit (default 2000)
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
- 发送第一条消息后，对话才会保存到 `data/folium.db`
- 切换到其他对话前，当前已有内容的对话会自动保存
- 每个会话同时保留完整历史和当前模型上下文：`messages.content` 保存原始内容，`messages.model_content` 只在内容被裁剪、压缩或注入 skill 后保存模型实际看到的版本；两者相同时只存一份。Web 历史展示使用完整历史，因此被裁剪的网页、PDF 或子 Agent 输出仍可恢复查看

## 工具

内置工具位于 `folium/tools/`：

```text
read_file       读取文件，支持 offset/limit
write_file      创建或覆盖文件（包括完整 LaTeX .tex 源文件）
edit_file       基于唯一字符串匹配的安全编辑（包括已有 LaTeX .tex 源文件），返回 diff
glob            按 glob 模式查找文件
grep            按正则搜索文件内容
bash            执行 shell 命令，支持 local/docker 后端、危险命令拦截、超时终止和输出截断
session_history 查询当前会话已持久化的完整历史；支持关键词搜索和按 message_id 分段读取
agent           启动子 Agent 处理独立子任务
todo            更新结构化任务列表，跟踪 pending / in_progress / completed
web_search      使用 Brave Search API 返回轻量 Web 搜索结果
web_fetch       读取单个 HTTP(S) URL，返回清洗后的 title 和正文片段
pdf_fetch       读取 PDF 正文文本
paper_search    通过 OpenAlex 搜索论文，返回结构化 JSON 证据
paper_validate  通过 OpenAlex 校验候选论文，标记 confirmed / partial / unverified / mismatch
```

`web_search` 是轻量搜索工具，只返回候选网页的 title、URL 和 snippet，不抓取全文、不做 RAG。需要设置 `BRAVE_SEARCH_API_KEY`；缺少 key 时工具会返回明确错误。

`web_fetch` 读取单个网页并返回清洗后的文本片段。它只允许 `http://` 和 `https://`，会拦截 localhost、内网地址、link-local 地址和重定向后的非公网地址；默认最多读取 2MB 响应体，最多返回 12000 字符，避免一次工具调用塞爆上下文。

`paper_search` 返回 OpenAlex 的结构化论文证据；`paper_validate` 用于最终输出前校验候选论文，未确认的文献应作为待核验线索，而不是已确认引用。

`todo` 工具用于长程、多步骤任务：

- Agent 会在系统提示词中要求多步骤任务使用 `todo`
- `todo` 每次接收完整列表替换当前状态，最多 20 项
- 状态只允许 `pending`、`in_progress`、`completed`
- 同一时间最多只能有一个 `in_progress`
- 如果连续 3 个工具调用轮次没有成功更新 todo，Agent 会向 messages 注入 `<reminder>Update your todos.</reminder>`
- Web 后端提供 `/todos`，并在 todo 更新时通过 SSE 发送 `todo_update`

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
Consecutive 5 tool call failures, current task stopped.
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

用户也可以在输入框使用 `/skill-name` 前缀直接激活某个 skill，例如 `/literature-review 帮我找关于 Transformer 的论文`，系统会自动注入该 skill 的完整内容。

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

Folium 采用预处理层加三级渐进式上下文压缩策略，自动触发时优先使用 LLM API 返回的 usage，并对新增内容使用本地估算：

| 层级 | 触发阈值 | 操作 | 成本 |
|------|---------|------|------|
| 预处理层 | 50% | 折叠完全相同且不少于 200 字符的 tool 输出：保留最新完整结果，将较早重复结果替换为占位符；豁免工具跳过 | 零 LLM 调用 |
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
- 预处理层会先对超过 50% 输入预算时出现的重复 tool 输出做精确去重，再进入后续可能有损的截断、占位符压缩和摘要流程；仅字符串内容、长度不少于 200 字符且不属于豁免工具的结果参与去重
- `bash`、`grep`、`glob` 等 primary 工具在 60% 后可裁剪；`read_file`、`agent` 等 secondary 工具在 70% 后可裁剪，但 80% 的占位符压缩会跳过 secondary 工具
- Layer 1 会跳过最近 2 个工具调用轮次的 tool 输出，避免刚读到的文件、实验结果或子 Agent 结论立刻被裁剪
- 上下文压缩只作用于发送给模型的 `messages`；用于 Web 历史和持久化复盘的 `transcript` 保留原始 user / assistant / tool 消息，因此完整工具输出不会因去重或其他压缩层丢失

费用计算：
- 支持缓存 token 单独计费（`prompt_cache_hit_tokens`）
- 费用公式：`(prompt - cached) × 输入价 + cached × 缓存价 + completion × 输出价`

## 本地可观测性

Folium 将会话与可观测性数据统一保存到本地 SQLite。一次用户输入会生成一个 trace；一次 LLM 调用、工具调用、Agent round、上下文压缩会生成对应 span 或 event。

默认保存位置：

```text
data/folium.db
```

数据库包含四张业务表：

- `sessions`：会话 ID、模型、system prompt、创建时间和更新时间
- `messages`：完整对话历史、模型实际使用的压缩上下文，以及工具调用关联信息
- `traces`：每次 Agent 执行的摘要，如会话、轮次、状态和耗时
- `trace_events`：trace 内的细粒度事件、span、快照、工具元数据和错误信息

其中 `messages` 保存会话内容，`traces` / `trace_events` 保存执行过程；两者用途不同，不会重复保存一份完整工具输出。工具输出的完整原文保存在 `messages.content`，trace 默认只保存脱敏后的预览、长度和 hash。

当前 trace 事件包括：

- `user_task`：一次用户输入触发的完整 Agent 执行
- `agent_round`：每轮 Agent 循环
- `llm`：模型调用、消息数量、工具数量、token、首 token 时间、输出摘要
- `llm_error`：模型调用失败时的 provider、状态码、错误类型、错误码和 request id
- `tool`：工具名称、参数、结果摘要、耗时、错误状态
- `sandbox_event`：Docker 工作区模式、容器启动、命令结束、超时和清理等生命周期事件
- `todo_update` / `todo_reminder`：todo 状态更新和自动提醒注入
- `context_compression`：上下文压缩前后的 token、消息数量和实际执行的压缩层
- `llm_request_snapshot` / `llm_response_snapshot`：每次模型调用实际输入和模型响应快照
- `context_snapshot`：上下文压缩前后的 `messages` 快照
- `agent_result`：最终回复摘要、消息数量和上下文 token 估算

可观测性配置：

```text
FOLIUM_OBSERVABILITY=1
FOLIUM_TRACE_MODE=all
FOLIUM_DB_PATH=data/folium.db
FOLIUM_TRACE_FULL_USER_INPUT=1
FOLIUM_TRACE_FULL_LLM_INPUT=0
FOLIUM_TRACE_FULL_LLM_OUTPUT=0
FOLIUM_TRACE_FULL_CONTEXT_SNAPSHOTS=0
FOLIUM_TRACE_FULL_TOOL_ARGS=1
FOLIUM_TRACE_FULL_TOOL_OUTPUT=0
FOLIUM_TRACE_REDACT_SECRETS=1
FOLIUM_TRACE_MAX_PREVIEW_CHARS=1000
```

默认 trace 只保存快照 preview、长度和 hash；打开 `FOLIUM_TRACE_FULL_LLM_INPUT`、`FOLIUM_TRACE_FULL_LLM_OUTPUT` 或 `FOLIUM_TRACE_FULL_CONTEXT_SNAPSHOTS` 后，会把对应完整内容写入 `trace_events`，适合调试但会显著增加数据库体积。

查看 trace：

```text
/traces
/trace <trace_id>
```

## 项目结构

```text
folium/
├── __main__.py              Web 入口
├── agent.py                 Agent 主循环、工具调用和观测插桩
├── llm.py                   OpenAI 兼容 LLM 客户端和 LiteLLM 后端
├── context.py               上下文估算与压缩
├── session.py               会话保存、读取、切换和删除
├── config.py                环境变量配置
├── prompt.py                系统提示词
├── role.md                  角色介绍，可自定义
├── observability/           本地 trace、span、脱敏和摘要读取
├── tools/                   内置工具
└── web/
    ├── server.py            FastAPI + SSE 后端
    └── static/index.html    Web 前端
```

## 科研智能体改造方向

后续计划围绕科研工作流和 Harness 组件继续扩展：

- 文献综述与研究方向发现
- 数学推导与公式验证
- 仿真实验代码生成与运行
- TeX 论文写作
- Agentic RAG 和证据链追踪
- 沙箱执行与文件影响追踪
- Artifact 记录，如报告、TeX、代码、图表、实验日志
- 评估与反馈机制
- Langfuse、Phoenix 或 OpenTelemetry 等外部观测集成

## Background memory maintenance

The Web server schedules a conservative background memory pass after the response is
fully sent. By default, it runs after 10 completed turns without main-agent memory use,
uses the same model as the main agent, and is limited to 5 model/tool rounds with a
2,000-token output budget. It receives a copy of the completed main-agent messages and
the same visible tool schemas, then appends a final English memory-maintenance user
prompt. Runtime execution remains memory-only: calls to any other visible tool are
rejected without side effects.

The pass may choose `NO_CHANGE`. It skips rather than truncates context when the copied
request plus its output budget would exceed the model context limit, and treats that
pass as checked. Its trace records the copied-context source, message and visible-tool
counts, approximate input tokens, rejected tool calls, final status, and cache-hit
tokens, without recording memory contents. Memory writes use version checks so a
concurrent main-agent write wins; the background task rereads and retries once, then
skips conflicts.

## 测试

当前可用的 unittest：

```bash
python -m unittest tests.test_tool_validation tests.test_tool_encoding tests.test_observability tests.test_web_server tests.test_session
```

如果安装了 pytest，也可以运行：

```bash
pytest
```

## 写入与审批

Web UI 中，`write_file` 和 `edit_file` 会直接执行，并在对应工具卡片中展示完成后的统一 diff。看起来会写入挂载工作区的 `bash` 命令仍会在执行前展示命令预览并等待用户审批；审批不会自动超时。`general` 子 Agent 调用 `write_file` 或 `edit_file` 时也直接执行并返回 diff。

## License

MIT。基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 二开，原作者何宇峰。
