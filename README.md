# 智能学习助手（Knowledge Agent）

面向学习场景的 **RAG + Agent** Demo：根据本地知识库回答，支持多步规划、工具调用与答案反思。

## 解决的问题

学习概念分散时，希望有一个助手能：

- 基于本地文档回答（减少胡编）
- 把复杂问题拆成步骤再执行
- 需要时调用计算器 / 查时间等工具
- 对最终答案做一次检查修订

## 功能

- 多轮对话（Streamlit 网页）
- 知识库问答（Embedding 向量检索 + 引用来源）
- Agent 流程：计划 → 逐步执行 → 汇总回答 → 反思修订
- 工具调用：计算器、当前时间（Function Calling）
- 清空对话

## 架构（简图）

```text
用户问题
  → make_plan          拆成 2~4 个短步骤
  → run_step × N       每步：检索知识 +（可选）工具调用
  → 汇总生成最终回答
  → reflect_answer     检查/修订后展示给用户
```

检索：`knowledge.txt` 切段 → embedding 预计算 → 问题向量与各段算余弦相似度 → Top-K。

## 技术栈

- Python
- Streamlit
- OpenAI SDK（兼容通义千问 / 百炼）
- Embedding：`text-embedding-v3` + 余弦相似度
- Function Calling

## 如何运行

1. 进入本目录，建议使用虚拟环境  
2. 安装依赖：`pip install -r requirements.txt`  
3. 设置环境变量（PowerShell 示例）：

```powershell
$env:qianwen_api="你的API密钥"
# 可选：聊天与 embedding 若地址不同，分别设置
$env:QIANWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

也可参考 `.env.example`（不要把真实 key 写进仓库）。

4. 启动：

```powershell
streamlit run app.py
```

## 我的思考

### 为什么用向量检索，而不是数相同字？

换一种问法时，关键词容易搜空。向量比较的是语义是否接近。

### 为什么要 Plan → Execute，而不是一次答完？

复杂问题容易漏点；拆步后每步可检索、可调工具，过程可展示，也方便排查错在哪一步。

### 为什么还要 Reflect？

汇总回答仍可能漏问、空泛或跑偏；再质检一轮，用确定性流程兜住「只靠一次生成」的风险。

### 为什么分 `api_messages` 和 `display_messages`？

发给模型的内容可含资料与工具结果；页面只显示简洁的「你 / AI」，并单独展示来源。

### 为什么工具调用常常要请求两次？

第一次让模型决定是否用工具；本地执行工具后，第二次根据真实结果再生成回答。

## 练习文件说明

`practice.py` 是学习过程中的积木练习（检索、计划、汇总、反思），正式演示以 `app.py` 为准。

## 下一步（可选）

- 小规模评测集（10～20 条 case）并记录改 prompt 前后效果
- 模块拆分：`rag.py` / `agent.py` / `tools.py`
