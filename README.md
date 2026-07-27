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
- Streamlit
- OpenAI SDK（兼容通义千问）

## 如何运行

1. 安装依赖：`pip install -r requirements.txt`
2. 设置环境变量 `qianwen_api` 为你的 API Key
3. 运行：`streamlit run app.py`

## 我的思考

（下一步填写：为什么分 api_messages / display_messages；工具为什么请求两次等）
