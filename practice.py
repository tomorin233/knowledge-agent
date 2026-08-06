import os
from openai import OpenAI
import json

import math


# 只从环境变量读密钥，不要把 key 写进代码
API_KEY = os.getenv("qianwen_api")
BASE_URL = os.getenv(
    "QIANWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not API_KEY:
    raise RuntimeError("请先设置环境变量 qianwen_api")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 聊天用上面的 maas 地址；向量模型常在百炼兼容地址上
# 若仍 404：到控制台确认已开通 text-embedding-v3，或换你控制台里显示的模型名
EMBEDDING_BASE_URL = os.getenv(
    "QIANWEN_EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
embedding_client = OpenAI(api_key=API_KEY, base_url=EMBEDDING_BASE_URL)


def format_steps(plan: list[str]) -> str:
    lines = ["计划步骤："]
    for step in plan:
        lines.append(f"- {step}")
    return "\n".join(lines)


def join_knowledge(related: list[str]) -> str:
    if not related:
        return "没找到相关知识"
    return "\n\n".join(related)


def make_rag_messages(question: str, knowledge: str) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": "你是助手。只能根据提供的知识回答。知识不足时明确说「根据已有知识无法回答。不要使用知识以外的常识、新闻或猜测",
        },
        {
            "role": "user",
            "content": f"知识：\n{knowledge}\n\n问题：{question}",
        },
    ]
    return messages


def ask_model(messages):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    return response.choices[0].message.content

def answer_with_knowledge(question: str,related: list[str]) -> str:
    knowledge= join_knowledge(related)
    messages = make_rag_messages(question, knowledge)
    return ask_model(messages)

def simple_search(question:str)->list[str]:
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text=f.read()

    chunks=[part.strip() for part in text.split("\n\n") if part.strip()]
    q=question.lower()
    keywords=["rag","检索","embedding"]
    results=[]
    for chunk in chunks:
        chunk_lower=chunk.lower()
        if any(k in chunk_lower for k in keywords) and any(k in q for k in keywords):
            results.append(chunk)
        if len(results)>=3:
            break
    return results

def answer(question: str)-> str:
    related=vector_search(question)
    if not related:
        return "我无法回答这个问题，因为我没有足够的知识。"
    return answer_with_knowledge(question,related)

def get_embedding(text_input: str) -> list[float]:
    # 注意：转向量用 embeddings.create，不是 chat.completions
    # 正确模型名一般是 text-embedding-v3（不是 qwen-embedding-v1）
    response = embedding_client.embeddings.create(
        model="text-embedding-v3",
        input=text_input,
    )
    return response.data[0].embedding

def cosine_similarity(vec_a:list[float],vec_b:list[float])->float:
    dot=sum(a*b for a,b in zip(vec_a,vec_b))
    norm_a=math.sqrt(sum(a*a for a in vec_a))
    norm_b=math.sqrt(sum(b*b for b in vec_b))
    if norm_a==0 or norm_b==0:
        return 0.0
    return dot/(norm_a*norm_b)

def vector_search(question:str, top_k:int=2)->list[str]:
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text=f.read()

    chunks=[part.strip() for part in text.split("\n\n") if part.strip()]
    q_vec=get_embedding(question)
    scored=[]
    for chunk,chunk_vec in zip(chunks,chunk_vectors):
        score=cosine_similarity(q_vec,chunk_vec)
        scored.append((score,chunk))
    scored.sort(key=lambda item:item[0],reverse=True)
    if not scored or scored[0][0]<0.35:
        return []
    return [chunk for score,chunk in scored[:top_k]]

def load_chunks(path:str="knowledge.txt")->list[str]:
    with open(path, "r", encoding="utf-8")as f:
        text=f.read()
    return [part.strip() for part in text.split("\n\n") if part.strip()]

def make_plan(question: str)-> list[str]:
    response=client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": ("你是助手。你的任务是把复杂问题拆成步骤。"
                            "只输出 JSON 数组，例如：[\"步骤1\", \"步骤2\"]。"
                            "不要输出其他文字。"
                            "每一步是动作（查找 / 总结 / 举例），不要写最终答案"
                            "每步尽量短，不超过 20 个字"
                            "仍然只输出 JSON 数组"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    text=response.choices[0].message.content
    plan = json.loads(text)
    return plan

def run_one_step(question:str,step:str)->str:
    related=vector_search(question +"\n"+step)
    if not related:
        return "本步没有找到相关知识"
    return answer_with_knowledge(f"{question}\n请完成步骤:{step}",related)

def run_plan(question:str)->list[str]:
    plan=make_plan(question)
    step_results=[]
    for step in plan:
        result=run_one_step(question,step)
        step_results.append(result)
    return step_results

def final_answer(question:str,step_results:list[str])->str:
    joined="\n\n".join(
        f"步骤{i+1}结果：{r}" for i,r in enumerate(step_results)
    )
    messages=[
        {
            "role":"system",
            "content":"你根据各步骤结果汇总成完整回答。不要编造步骤里没有的信息。"
        },
        {
         "role":"user",
         "content": f"用户问题：{question}\n\n{joined}\n\n请给出最终回答"
        }

    ]
    return ask_model(messages)

def reflect_answer(question:str,answer:str)->str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是质检员。检查回答是否完整覆盖用户问题、是否空洞或自相矛盾。"
                "若有问题就给出改正后的完整回答；若没问题，原样输出回答。"
                "不要编造知识库中没有依据的内容。"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{question}\n\n待检查回答：{answer}\n\n请输出最终回答：",
        },
    ]
    return ask_model(messages)

def agent_answer(question:str)->str:
    step_results=run_plan(question)
    draft=final_answer(question,step_results)
    return reflect_answer(question,draft)

chunks=load_chunks()
chunk_vectors=[get_embedding(chunk) for chunk in chunks]


print(agent_answer("根据知识库解释什么是RAG，并举一个使用场景"))