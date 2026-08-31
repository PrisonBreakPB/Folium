---
name: paper-drafting
description: 在用户明确要求撰写完整论文(.tex)时使用。处理从"文献调研结果 + 理论草稿 + 实验图"三类材料到可编译 IEEE 论文初稿的流程。模板与风格规范见下方引用的外部文件。
---

# Paper Drafting Workflow（论文初稿流程）

把用户手上的三类材料，映射成一篇可编译的 IEEE 论文初稿。

## 材料来源 → 论文章节

| 用户提供的材料 | 供应部分 | 说明 |
|---|---|---|
| 文献调研结果（如 `control-literature-search` 产物） | Introduction | 讲"研究领域的发展"，引背景与相关方向 |
| 理论草稿（本类论文重心，含数学推导） | Problem Formulation + Main Result | 把自由推导规范化成定理/证明、统一符号 |
| 实验代码跑出的图片（png/pdf） | Simulation | 插图 + caption + \ref |

## 步骤

1. **定位三类材料**：先问清/找见输入（调研结果文件、理论草稿路径、实验图目录），不臆测。
2. **定模板 + 搭骨架**：复制 `template/paper.tex`（IEEEtran journal 框架）为 `main.tex`，一次 `write_file` 落出段落占位。不要改模板本身。
3. **Introduction**：读调研结果，写领域发展/动机/贡献列表，配 `\cite{}` 与 `.bib`。
4. **Main Result（重心）**：从理论草稿提炼——统一符号表 → 定义问题 → 按顺序整理 theorems/lemmas → 每个配 proof。数学推导保留证明，不压缩成结论摘要；公式编号一致，`\label`/`\ref` 贯通。
5. **Simulation**：把实验图嵌成 `figure + caption + \ref`，先写每个图要支撑的结论再放图。
6. **引用归一**：把 `paper_search`/`paper_validate` 返回的 DOI/作者对进 `.bib`，正文 `\cite{}`；不新造条目。
7. **编译验证**：在 **host**（非 docker 沙箱，沙箱无 TeX）用 `latexmk -pdf main.tex` 或 `pdflatex main.tex`（两遍）确认零错误。
8. **复查**：无残留占位、符号一致、图引用对得上、编译通过。

## 分步写法要点

- **Main Result**：符号表 → Problem statement → 主定理（先给结论）→ 前置引理 → 逐步证明。理论草稿常用 prose/推导，需规范成 theorem/lemma/proof 环境，不照抄原文排版。
- **Simulation**：图放 caption + \label，正文用 \ref 引用；图与结果的对应关系写清楚。

## 遵守

- 语言与用词规范：动笔前若存在 `docs/paper-style.md` 则 `read_file` 它（禁用词、时态、简洁要求）。
- 学术诚信：引用只出自工具返回元数据，不编造 DOI/作者/年份。
- LaTeX 边界：仅在用户明确要求写论文时走本流程；若用户仅让检查/编译/给意见，直接照常规处理，不使用本 skill。

## 参考文件

- 模板框架：`template/paper.tex`
- 语言/风格规范（可选）：`docs/paper-style.md`