# 智能学习助手

## 解决的问题

学习概念分散时，希望有一个助手能基于本地知识库回答，并支持计算、查时间等工具。

## 功能

- 多轮对话
- 知识库问答（RAG）
- 工具调用：计算器、当前时间
- 清空对话
- Streamlit 网页界面

## 技术栈
- Python
- Streamlit（网页界面）
- OpenAI SDK（兼容通义千问）
- Embedding 向量检索（text-embedding-v3 + 余弦相似度）
- Function Calling（计算器、当前时间）

## 如何运行

1. 安装依赖：`pip install -r requirements.txt`
2. 设置环境变量 `qianwen_api` 为你的 API Key
3. 运行：`streamlit run app.py`

## 我的思考
### 为什么用向量检索，而不是数相同字？
相同字匹配遇到换一种问法容易搜不准。
向量检索比较的是「意思像不像」，例如「什么是变量」和「变量是什么意思」也能对上。
### 为什么分 api_messages 和 display_messages？
发给模型的消息需要带资料和工具结果；
网页上只显示简洁的「你 / AI」，避免把整段资料刷在界面上。
### 为什么工具调用要请求两次？
第一次让模型决定要不要用工具；
Python 真正执行工具后，第二次再根据真实结果生成最终回答。

