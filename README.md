# Folium

> 基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 二开 —— 一个极简 AI 编程 Agent 蓝图。


[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) | [中文](README_CN.md)

**CoreCoder 的二开版本（~1,400 行 Python）。**

CoreCoder 把 Claude Code 的关键架构模式浓缩在 ~1,400 行 Python 里。Folium 是在此基础上进行的二开项目。

---

```
$ folium -m deepseek-chat

You > 读一下 main.py，修掉拼错的 import

  > read_file(file_path='main.py')
  > edit_file(file_path='main.py', ...)

--- a/main.py
+++ b/main.py
@@ -1 +1 @@
-from utils import halper
+from utils import helper

修好了：halper → helper。
```

## 你能得到什么

| 设计模式 | Claude Code | Folium |
|---|---|---|
| 搜索替换编辑（唯一匹配 + diff） | FileEditTool | `tools/edit.py` — 70 行 |
| 并行工具执行 | StreamingToolExecutor（530行） | `agent.py` — ThreadPool |
| 三层上下文压缩 | HISTORY_SNIP → Microcompact → CONTEXT_COLLAPSE | `context.py` — 145 行 |
| 子代理隔离上下文 | AgentTool（1,397行） | `tools/agent.py` — 50 行 |
| 危险命令拦截 | BashTool（1,143行） | `tools/bash.py` — 95 行 |
| 会话持久化 | QueryEngine（1,295行） | `session.py` — 65 行 |
| 动态系统提示词 | prompts.ts（914行） | `prompt.py` — 35 行 |

## 安装

```bash
pip install -e .
```

选你的模型，任何 OpenAI 兼容 API 都行。可以 `export` 环境变量，也可以在项目根目录放一个 `.env` 文件：

```bash
# DeepSeek V3
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com
folium -m deepseek-chat

# OpenAI GPT-5
export OPENAI_API_KEY=sk-...
folium -m gpt-5

# Qwen 3.5
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
folium -m qwen-max

# Ollama（本地）
export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1
folium -m qwen3:32b

# 单次模式
folium -p "给 parse_config() 加上错误处理"
```

## 架构

整个项目一目了然：

```
folium/
├── cli.py            REPL + 命令                   218 行
├── agent.py          Agent 循环 + 并行执行          122 行
├── llm.py            流式客户端 + 重试              156 行
├── context.py        三层压缩                       196 行
├── session.py        会话保存/恢复                   68 行
├── prompt.py         系统提示词                      33 行
├── config.py         环境变量配置                    55 行
└── tools/
    ├── bash.py       Shell + 安全 + cd 追踪         115 行
    ├── edit.py       搜索替换 + diff                  85 行
    ├── read.py       文件读取                         53 行
    ├── write.py      文件写入                         36 行
    ├── glob_tool.py  文件搜索                         47 行
    ├── grep.py       内容搜索                         78 行
    └── agent.py      子代理生成                       58 行
```

## 当库用

```python
from folium import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="your-key", base_url="https://api.deepseek.com")
agent = Agent(llm=llm)
response = agent.chat("找出项目里所有 TODO 注释并列出来")
```

## 加自定义工具（约 20 行）

```python
from folium.tools.base import Tool

class HttpTool(Tool):
    name = "http"
    description = "请求一个 URL。"
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def execute(self, url: str) -> str:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()[:5000]
```

## 命令

```
/model           查看当前模型
/model <名称>    切换模型
/compact         压缩上下文（对标 Claude Code 的 /compact）
/tokens          查看 token 用量 + 费用估算
/diff            查看本次会话修改的文件
/save            保存会话
/sessions        列出已保存的会话
/reset           清空历史
quit             退出
```

保存的会话 ID 会先安全化再作为文件名，恢复数据始终留在 `~/.folium/sessions` 目录内。

## License

MIT。基于 [CoreCoder](https://github.com/he-yufeng/CoreCoder) 二开，原作者何宇峰。