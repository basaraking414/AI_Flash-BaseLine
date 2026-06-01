# AI Flash Portrait - 项目级 CLAUDE.md

## 核心行为准则

- **Think Before Coding**
  - 不确定时先问，不要擅自假设。不要隐藏困惑，应主动呈现权衡。
  - 实施前：明确说出你的假设。如果存在多种解读，全部列出，不要默默选择一个。如果有更简单的方法，说出来。遇到不清楚的地方，停下来并指出困惑之处。

- **Simplicity First**
  - 用最少的代码解决问题，拒绝投机性开发。
  - 不添加未被要求的功能。不为只用一次的代码创建抽象。不添加无人要求的"灵活性"或可配置性。不处理不可能发生的异常场景。
  - 如果你写了 200 行而它能是 50 行，请重写它。问问自己："一个高级工程师会觉得这段代码过度复杂吗？" 如果是，就必须简化。

- **Surgical Changes**
  - 只触碰你必须触碰的代码，只清理你造成的混乱。
  - 修改现有代码时：不要"顺手"改进无关的代码、注释或格式。不要重构没坏的东西。匹配现有的风格。如果你发现了无关的死代码，只提一嘴，不要删除它。
  - 如果改动造成孤立代码，清除你自己引入的孤立导入 / 变量 / 函数。未经要求不要删除预先存在的无效代码。
  - 检验标准：每一行修改都应直接追溯到用户的请求。

- **Goal-Driven Execution**
  - 明确定义成功标准，并循环直至验证。
  - 将模糊任务转化成可验证的目标：
    - "添加验证" → "为无效输入写测试，然后让测试通过"
    - "修复 Bug" → "编写能复现该 Bug 的测试，然后让测试通过"
    - "重构 X" → "确保重构前后的测试都能通过"
  - 对于多步骤任务，陈述简要计划，每个步骤附上验证点：
    - [步骤] → 验证: [检查]
  - 强有力的成功标准可以让 AI 独立循环优化。

## 工作流规则 — Skill 自动触发

触发 skill 时，必须在回复开头显示：`[自动激活] <skill名称> — <原因简述>`

### Superpowers 插件技能（自动管理触发）

- brainstorming
- writing-plans
- executing-plans
- test-driven-development
- systematic-debugging
- verification-before-completion
- requesting-code-review
- receiving-code-review
- subagent-driven-development
- using-superpowers
- writing-skills

### 非 Superpowers 技能触发规则

#### find-skills — 搜索现成工具

**当出现以下信号时激活：**
- "有没有现成的 Skill"、"搜索技能"、"找一下…工具"
- "有没有工具能…"、"能不能自动化…"

#### skill-creator — 创建新技能

**当出现以下信号时激活：**
- "创建一个新技能"、"封装这个流程"、"把这个变成 Skill"
- "写一个 claude Skill"、"注册一个命令"

**创建完成后，必须询问用户：** "是否需要将新 Skill 的信息同步到全局 `CLAUDE.md` 中？"

#### paperbanana-diagram — 论文流程图生成

**当出现以下信号时激活：**
- "paperbanana"、"生成流程图"、"CVPR图表"、"架构图"
- "作methodology图"、"methodology图"、"论文作图"

**核心工作流：**
1. 执行前先向用户确认 3 个参数：渲染方式、迭代次数、目标评分
2. 读取项目代码，构建 source_context
3. 调用 PaperBanana 多 agent 管线，score_threshold 达标自动停止
4. 输出到 `figure/` 目录

## 项目数据存储规则

- 所有对话记录、计划（plans）、项目记忆（memory）**必须**保存在当前项目内的`.claude/`，禁止存入全局 `~/.claude/`
- 存储路径：
  - 对话：`.claude/conversations/`
  - 计划：`.claude/plans/`
  - 记忆：`.claude/memory/`
  - 临时文件：`.claude/scratch/`（不提交 Git）
- 若目录不存在则自动创建
- 当全局目录有相同内容时，以项目内为准

## 项目特定信息

### 项目简介

AI Flash Portrait 是一个基于 Retinex 分解的闪光肖像处理项目，使用 PyTorch 和 BasicSR 框架。

### 核心文件路径

- **模型架构**：`basicsr/models/archs/my_restormer_arch.py`
- **训练配置**：`configs/` 目录下的 YAML 文件
- **推理脚本**：`run_inference.py`
- **训练脚本**：`basicsr/models/` 目录下的训练相关文件

### 关键技术

- Retinex 分解：`I = R \times L`（反射图 × 光照图）
- 损失函数：L1 Loss、SSIM Loss、Perceptual Loss
- 框架：BasicSR

## Memory 系统

- 记忆文件存储在 `.claude/memory/` 目录
- `MEMORY.md` 是索引文件，每条记忆一行
- 记忆类型：user、feedback、project、reference
