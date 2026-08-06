import json
import os
from datetime import datetime
import math
import streamlit as st
from openai import OpenAI

# 项目根目录（保证无论从哪启动都能找到 knowledge.txt）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file(path: str) -> None:
    """读取 .env 到环境变量（不依赖 python-dotenv）。"""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig 避免记事本 BOM
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("qianwen_api")
BASE_URL = os.getenv(
    "QIANWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not API_KEY:
    st.error(
        "未检测到 qianwen_api。请确认 knowledge_agent 目录下有 .env，"
        "且第一行为：qianwen_api=你的密钥（不要多余空格/引号）"
    )
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

def make_plan(question: str) -> list[str]:
    """让模型把复杂问题拆成步骤。"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是任务规划助手。把用户问题拆成 2~4 个简短步骤。"
                    "只输出 JSON 数组，例如："
                    '["步骤1", "步骤2", "步骤3"]'
                    "不要输出其他文字。"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    text = response.choices[0].message.content or "[]"
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        plan = json.loads(text)
        if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
            return plan
    except Exception:
        pass
    return ["直接回答用户问题"]


def run_step(step: str, question: str) -> str:
    """执行计划中的一步：复用 search_knowledge / tools / run_tool / client。"""
    related = search_knowledge(f"{question}\n{step}", top_k=2)
    if related:
        knowledge_text = "\n\n".join(related)
    else:
        knowledge_text = "没有找到相关知识"

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "你在执行多步任务中的一个步骤。"
                    "只完成当前步骤，回答要简短。"
                    "需要计算就调用 calculator，需要时间就调用 get_current_time。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户原问题：{question}\n"
                    f"当前步骤：{step}\n"
                    f"参考资料：\n{knowledge_text}"
                ),
            },
        ],
        tools=tools,
    )
    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content or "（本步无输出）"

    # 复用 run_tool：先记下调用意图，再执行工具，再二次请求
    messages = [
        {
            "role": "system",
            "content": "根据工具结果，用一句话完成本步骤。",
        },
        {
            "role": "user",
            "content": f"原问题：{question}\n当前步骤：{step}",
        },
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
        },
    ]
    for call in assistant_message.tool_calls:
        tool_result = run_tool(call.function.name, call.function.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_result,
            }
        )

    second = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    return second.choices[0].message.content or "（本步无输出）"


def reflect_answer(question: str, answer: str) -> str:
    """对最终答案做一次检查/修订，只返回给用户看的正文。"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是质检员。检查回答是否完整覆盖用户问题、是否空洞或自相矛盾。"
                    "若有问题就给出改正后的完整回答；若没问题，原样输出回答。"
                    "只输出给用户看的最终回答正文，不要输出检查说明、评分或「无需修改」等话。"
                    "不要编造没有依据的内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"待检查回答：{answer}\n\n"
                    "请输出最终回答："
                ),
            },
        ],
    )
    return response.choices[0].message.content or answer


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
        if msg.get("sources"):
            st.caption("来源："+"；".join(msg["sources"]))

user_input = st.text_input("请输入问题", key="user_box")

if st.button("发送"):
    if not user_input.strip():
        st.warning("请先输入问题")
    else:
        st.session_state.display_messages.append(
            {"role": "user", "content": user_input}
        )

        plan = make_plan(user_input)
        st.info("计划步骤：\n- " + "\n- ".join(plan))

        step_results = []
        with st.spinner("正在按步骤执行..."):
            for step in plan:
                result = run_step(step, user_input)
                step_results.append(f"{step} → {result}")
                st.write(f"**步骤完成：** {step} → {result}")

        # 来源仍用原问题检索一次，方便展示 citation
        related = search_knowledge(user_input)
        source_titles = [chunk.split("\n", 1)[0] for chunk in related]

        steps_text = "\n".join(step_results)
        st.session_state.api_messages.append(
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_input}\n\n"
                    f"已完成的步骤结果：\n{steps_text}\n\n"
                    "请根据以上步骤结果，给出最终完整回答。"
                    "不要重复罗列步骤，直接给用户看的答案。"
                ),
            }
        )

        with st.spinner("正在生成最终回答..."):
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=st.session_state.api_messages,
            )
            draft = response.choices[0].message.content or ""

        with st.spinner("正在反思与修订..."):
            answer = reflect_answer(user_input, draft)

        st.session_state.api_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.display_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_titles,
            }
        )

        st.rerun()
