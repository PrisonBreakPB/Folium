# 记忆系统迁移方案 — MemoryTool → 通用文件工具

> 状态：**草案，待审**　适用版本：当前 main 分支
>
> 目标对齐 Claude Code 2.1.88 的 auto-memory 设计，但**精简、不完整照搬**，贴合 Folium 科研场景与"小步走"风格。

---

## 1. 背景与动因

当前 Folium 记忆系统是**单文件 + 专用 MemoryTool**：

- 记忆文件 `MEMORY_FILE` 硬编码为绝对路径 `D:\learn-Agent\Folium\memory.md`（`folium/prompt.py:7`），与会话、项目、启动目录完全无关
- 主 agent 通过 `MemoryTool`（`folium/tools/memory.py`）的 `read/append` 读写，`append` 带乐观锁版本号（SHA-256）、查重、结构约束
- 后台 `MemoryAgent`（`folium/memory_maintenance.py`）是受限 runner，只暴露 memory 工具

本次改造的目标：

1. 读写途径改为**通用文件工具**（FileRead / FileWrite / FileEdit），`MemoryTool` 退出
2. 记忆按 **git root 分项目**，落在全局用户目录而非项目工作目录
3. 记忆按 **3 类分文件**（`user / feedback / project`），不再是一个 memory.md
4. 解除 edit/write 工具对 memory.md 的写入禁止
5. 后台记忆 agent 增加**路径 clamp**，写权限锁死在记忆目录内
6. 主 agent、后台 agent 的提示词**借鉴** Claude Code，精简选用

---

## 2. 目标架构

### 2.1 存储布局

```
<home>/.folium/projects/<git-slug>/
├── user.md          # 用户偏好 / 角色 / 职责 / 合作方式
├── feedback.md      # 方法论纠正与成功确认
└── project.md       # 项目背景 / 决策 / 未决事项
```

- `<git-slug>`：从启动目录**沿目录树向上找最近的 `.git`**（纯文件系统扫描，不调用 git 命令）作为项目边界，得到 canonical git root，再卫生成 slug（对齐 Claude Code：`statSync` 向上找 `.git` 文件/目录）；**非 git 目录 fallback 用启动 cwd** 生成 slug
- 同一 git 仓库从任何子目录启动，均归同一项目（worktree 归一到主仓库根，本方案先不做 worktree 归一化，如需可后补）
- 不同 git 仓库、以及非 git 目录彼此隔离；同一仓库从任何子目录启动都归同一项目
- 目录由 harness 首次进入时预创建（不用 agent mkdir）

### 2.2 记忆分类（3 类）

| 文件 | 内容 | 对应 Claude Code |
|------|------|------------------|
| `user.md` | 用户是谁、角色、职责、知识、合作偏好 | `user` |
| `feedback.md` | 方法论指导：纠正（"别这样做"）+ 成功确认（"这种方式可行"） | `feedback` |
| `project.md` | 当前项目在做什么、决策、背景、未决事项 | `project`（reference 并入） |

每类**建议**（提醒后端已定，不强制）配 `when_to_save / what_to_save / what_not_to_save`。

### 2.3 文件内部组织（纯 Markdown）

每条记忆：

```
### <一句话标题>

<正文，可多行>

---
```

- 用标准 Markdown `###` 标题 + `---` 分隔线，模型解析边界最可靠
- **不引入 frontmatter / YAML**（Q8 定案：先不做元数据，为后续 Agentic RAG 留口子但当下不欠）

---

## 3. 代码落地清单

### 3.1 存储路径（替换硬编码）

**`folium/prompt.py`**
- 删除模块级硬编码：
  ```python
  MEMORY_FILE = Path(r"D:\learn-Agent\Folium\memory.md")
  ```
- 改为从项目上下文计算路径的函数（如 `get_memory_dir()` / `get_memory_files()`），返回 3 文件的路径
- 由启动流程基于 git root 初始化并预创建目录

**`folium/config.py`**
- 增加配置项：记忆根目录（默认 `<home>/.folium/projects`）、是否启用

**git root 探测（对齐 Claude Code `findCanonicalGitRoot`）**
- 新增工具函数：从启动 cwd 沿目录树向上 `statSync` 找最近的 `.git` 文件/目录 → 即项目边界；找不到则返回 None
- 项目边界 = `找.git得到的root ?? 启动cwd`（非 git 目录 fallback 到 cwd）
- **纯文件系统扫描，不调用 git 命令**（Claude Code 用的是 `statSync` walk，`git rev-parse` 只用于分支检测；本方案沿用，避免依赖 git 二进制、更可移植）
- 注意：Windows 盘符/分隔符、子目录启动、非 git 目录

**slug 卫生化（对齐 Claude Code `sanitizePath`）**
- `name.replace(/[^a-zA-Z0-9]/g, '-')`：所有非字母数字字符（含盘符冒号、路径分隔符）替换成 `-`，**保留前导 `/` 和连续 `-`**
- 长度 ≤ 200 直接返回；超过 200 截断并追加 hash 后缀防碰撞
- 示例：`D:\learn-Agent\Folium` → `-D--learn-Agent-Folium`

### 3.2 解除 write/edit 禁止

**`folium/tools/write.py:23`**、**`folium/tools/edit.py:35`**
- 当前写死 `"Never use this tool to modify memory.md"`（强制只能走 memory 工具）
- **一并移除**，让 FileWrite / FileEdit 可以正常落记忆文件

### 3.3 MemoryTool 退出

**`folium/tools/memory.py`**、**`folium/tools/__init__.py`**
- `MemoryTool` 及注册移除（主 agent 不再需要专用记忆工具）
- 其提供的乐观锁/查重职责转移到别处（见 4.5 并发与去重）

### 3.4 后台记忆 agent 改为文件工具 + 路径 clamp

**`folium/memory_maintenance.py`**
- `MemoryAgent._run_loop`：执行单步从 `_execute_memory_call(tool_call)` 改为"通用文件工具，套路径 clamp"
- 新增工具级认知白名单（对齐 Claude Code `createAutoMemCanUseTool`）：

  | 工具 | 权限 |
  |------|------|
  | FileRead / Grep / Glob | 放行（不限目录，后台要读对话上下文） |
  | Bash | 仅只读（ls/find/grep/cat/stat/wc/head/tail） |
  | FileEdit / FileWrite | **仅当 `file_path` 落在当前项目记忆目录内** |
  | Bash rm / 写 Bash | deny |
  | 其余（Agent/MCP 等） | deny |

- 当前"只暴露 memory 工具"的受限 runner 形态升级为"通用文件工具的路径 clamp"形态，安全目标一致（写锁死记忆目录），但能读上下文做判断

**`folium/memory_maintenance.py`** 调度触发逻辑同步调整（见 3.5）

### 3.5 后台触发：保留主/后台互斥，改为扫描文件工具+路径

**`folium/memory_maintenance.py`** / CLI / Web 接入
- 现状：`on_turn_completed` 判 `main_agent_used_memory`（扫描 transcript 里 `name == "memory"`）来决定是否重置轮数
- 改为：扫描主 agent 本轮是否用 **FileWrite / FileEdit 且路径落在当前项目记忆目录内**（对齐 Claude Code `hasMemoryWritesSince`）
- 命中了 → 跳过/重置轮数（主 agent 已写，后台兜底冗余）；否则每 N 轮触发
- **重要**：只匹配**当前项目**的记忆路径，不误扫其它项目的记忆写

### 3.6 阈值

延续 `FOLIUM_MEMORY_MAINTENANCE_TURNS`（当前默认 5），单点定义于 `config.py` 的 `DEFAULT_MEMORY_MAINTENANCE_TURNS`。

---

## 4. 提示词设计

### 4.1 主 agent 系统提示词记忆段（`folium/prompt.py` `_memory_section`）

借鉴 Claude Code，精简选取，不整段照搬。要点：

- 目录已存在、用文件工具直接读写（not mkdir）
- 记忆按类分 3 文件；每种类型一句话触发时机
- 如何保存：写条目到对应文件（`### 标题` + 正文 + `---`）
- 何时不保存：临时任务、代码实现细节、可从项目推导的东西、密钥
- 明确"记忆在 `<home>/.folium/projects/<git-slug>/`，与当前项目绑定，不是全局"

拟定草稿（正式措辞在方案落地时敲定）：

```markdown
## Long-Term Memory

You have a persistent, file-based memory split into 3 files under your project's memory
directory: `<home>/.folium/projects/<git-slug>/{user,feedback,project}.md`.

- `user.md` – who the user is, their role, preferences, how they like to collaborate
- `feedback.md` – methodological corrections ("don't do X") and confirmed approaches ("this worked")
- `project.md` – what this project is, decisions made, open items

Save a memory as a `### <title>` entry + body, separated by `---`. Update existing
entries instead of duplicating. Do not store secrets, one-off task commands, or things
derivable from the project code. If the user explicitly asks you to remember something,
save it immediately.
```

### 4.2 后台记忆 agent 系统提示词（参考 Claude Code，分两类、精简）

参考 Claude Code，后台拆成**两个可选职责**，可只做其一：

**A) 提取（frequent，参照 Claude Code extractMemories / 现有 `MEMORY_MAINTENANCE_USER_PROMPT`）**
- 约束式措辞：*"把前面的对话当素材审阅，不是继续执行原任务"*
- 只准在记忆目录内写，其余写操作会被拒绝
- 结束时返回状态 token（`NO_CHANGE / UPDATED / …`）

**B) 巩固（infrequent，参照 Claude Code autoDream consolidation）**
- 低频（如每 ≥24h 或 ≥N 轮）
- 去重、解决矛盾、压缩过期条目、保持索引精简
- 同样路径 clamp

> 可选：若不做 B，则仅靠 A + "更新已有条目而非新增"来缓解堆积。

---

## 5. 安全与取舍

| 关注点 | 结论 |
|--------|------|
| 后台写权限 | **写锁死记忆目录，读放开**——能读上下文，但写不进项目任何位置 |
| 单文件并发 | 3 文件结构天然分散写目标；同一文件并发仍存在 → 后台去重更新缓解（4.5） |
| 密钥/敏感 | 提示词明确不存；记忆文件在全局目录，仍需警惕（不进项目 git） |
| 挂起回收 | 现状无超时回收（`running` 可能永久卡死）；建议顺带补看门狗 |

### 4.5 并发与去重（承接 3.3 移除的乐观锁）

改造后无 MemoryTool 的 SHA-256 版本号。靠：

1. **分文件**降低写冲突面（3 个独立文件）
2. 后台与主 agent **互斥**（3.5）错开写入时机
3. 后台**读后去重/更新**而非只追加（4.2 提示词约束）
4. 若需严格一致性，可对单文件写加**进程内锁**（asyncio.Lock + 文件锁）——**可选**，不默认做

### 4.6 后台职责定案

- **只做"提取"**：保持现有行为——后台在每 N 轮触发时从对话中提取稳定事实，去重/更新写进 3 文件
- **autoDream（巩固/低频清理）本轮不做**：不单独引入低频 consolidation agent；堆积/过期靠"更新已有条目"缓解，后续如需再补

### 4.7 挂起回收（看门狗）

现有调度器 `_SchedulerState.running` 无超时回收：若后台任务因 LLM 卡住不返回，`finally` 不执行，`running` 永久为 True，该 session 再也无法触发记忆维护。

**需补**：为 `on_turn_completed` 触发的后台任务加超时/看门狗——参考 Claude Code 的 `.consolidate-lock` + PID 复用防护 + `HOLDER_STALE_MS`。落地方式二选一（可后定）：
- 在任务上设总超时（如 `asyncio.wait_for` 一个上限时长，超时即重建 running 状态）
- 后台任务完成记录的 `running` 若超过阈值未复位，下一轮触发时视为"过期"强制回收

---

## 6. 验证方式

- 单测：路径/git-slug 探测、文件工具 clamp、后台触发互斥（扫描文件工具+路径）、3 类文件读写
- 集成：CLI 从 git 仓库目录启动 → 记忆落在 `<home>/.folium/projects/<slug>/`；非 git 目录 fallback 行为
- 冒烟：主 agent 主动写记忆、后台触发、写权限被 clamp 拒绝的用例
- 现有测试：`tests/test_memory_tool.py`、`tests/test_memory_maintenance.py` 需随改造同步更新

---

## 7. 未决 / 待你确认

全部结清。

已定：
- 后台只做提取，不做巩固/autoDream（4.6）
- 需补挂起回收看门狗（4.7）
- 旧 `§` 记忆不迁移，一刀切 Markdown
- git-slug 卫生化规则（对齐 Claude Code `sanitizePath`）：取项目边界路径（找.git的root ?? cwd）→ 非 `[a-zA-Z0-9]` 全替换为 `-`，保留前导/连续 `-` → ≤200 直接用，超长截断+hash。示例 `D:\learn-Agent\Folium` → `-D--learn-Agent-Folium`
- 项目边界判定：纯文件系统向上找 `.git`（不跑 git 命令），非 git fallback 到 cwd

无未决项。