import json
import os
from datetime import datetime
import math
import streamlit as st
from openai import OpenAI

# 项目根目录（保证无论从哪启动都能找到 knowledge.txt）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.getenv(
    "qianwen_api",
    "sk-ws-H.EMHRREM.Uyvm.MEUCIQDOJhH-7nUUvPlLlx0I--QtPVk_6MEaBVDVAgx2-KT8kgIgOcEb2vMPbtHmTUAlOdp61hbvEUUUMGFkkds1VbuASbY"
)
BASE_URL = os.getenv(
    "QIANWEN_BASE_URL",
    "https://llm-nfrbx1834flhn3ix.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

if not API_KEY:
    st.error("未检测到环境变量 qianwen_api，请先设置 API Key 后再运行。")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

with open(os.path.join(BASE_DIR, "knowledge.txt"), "r", encoding="utf-8") as f:
    text = f.read()

chunks = [part.strip() for part in text.split("\n\n") if part.strip()]

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "一个计算器函数，用于执行数学计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return "错误：表达式只能包含数字和 + - * / ( )"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"错误：{e}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
}


def run_tool(tool_name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    func = TOOL_FUNCTIONS[tool_name]
    if tool_name == "calculator":
        return func(args["expression"])
    return func()

def get_embedding(text_input: str) -> list[float]:
    # 注意：转向量要用 embeddings，不是 chat.completions
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=text_input,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@st.cache_resource
def build_chunk_vectors():
    vectors = []
    for chunk in chunks:
        vectors.append(get_embedding(chunk))
    return vectors


# 必须写在 get_embedding 定义之后
chunk_vectors = build_chunk_vectors()


def search_knowledge(question: str, top_k: int = 2) -> list[str]:
    question_vector = get_embedding(question)
    scored = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(question_vector, chunk_vector)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


SYSTEM_PROMPT = (
    "你是学习助手。优先根据资料回答；资料没有就说不知道。"
    "计算题用 calculator，时间问题用 get_current_time。"
)


def reset_chat():
    st.session_state.api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    st.session_state.display_messages = []


st.title("智能学习助手")
st.write("多轮对话 + 知识库问答 + 工具调用")

if "api_messages" not in st.session_state:
    reset_chat()

if st.button("清空对话"):
    reset_chat()
    st.rerun()

for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        st.markdown(f"**你：** {msg['content']}")
    else:
        st.markdown(f"**AI：** {msg['content']}")

user_input = st.text_input("请输入问题", key="user_box")

if st.button("发送"):
    if not user_input.strip():
        st.warning("请先输入问题")
    else:
        st.session_state.display_messages.append(
            {"role": "user", "content": user_input}
        )

        related = search_knowledge(user_input)
        if related:
            knowledge_text = "\n\n".join(related)
        else:
            knowledge_text = "没有找到相关知识"

        st.session_state.api_messages.append(
            {
                "role": "user",
                "content": (
                    f"资料：{knowledge_text}\n\n"
                    f"问题：{user_input}"
                ),
            }
        )

        with st.spinner("AI 思考中..."):
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=st.session_state.api_messages,
                tools=tools,
            )
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                answer = assistant_message.content or ""
                st.session_state.api_messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.session_state.api_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in assistant_message.tool_calls
                        ],
                    }
                )
                for call in assistant_message.tool_calls:
                    tool_name = call.function.name
                    tool_result = run_tool(tool_name, call.function.arguments)
                    st.info(
                        f"工具：{tool_name}({call.function.arguments}) -> {tool_result}"
                    )
                    st.session_state.api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": tool_result,
                        }
                    )
                second_response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=st.session_state.api_messages,
                )
                answer = second_response.choices[0].message.content or ""
                st.session_state.api_messages.append(
                    {"role": "assistant", "content": answer}
                )

            st.session_state.display_messages.append(
                {"role": "assistant", "content": answer}
            )

        st.rerun()
