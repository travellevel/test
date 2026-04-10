import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, MessagesState
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOllama(model="deepseek-r1:7b")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

SYSTEM_PROMPT = """
You are a research assistant. You have access to the following tools:
- search_tool(query): Search the web for information.
- wiki_tool(query): Query Wikipedia for information.
- save_tool(data, filename="research_output.txt"): Save research data to a file. After you have the final answer, you MUST call save_tool to save the final JSON data.

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search_tool, wiki_tool, save_tool]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, wrapped in a JSON with fields: topic, summary, sources, tools_used

After Final Answer, you must call save_tool to save the JSON data.
Wrap the final answer in the exact JSON format:
{format_instructions}
"""

tools = [search_tool, wiki_tool, save_tool]
tool_map = {tool.name: tool for tool in tools}

def parse_action(text: str):
    """解析模型输出中的 Action 和 Action Input"""
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if action_match and input_match:
        return action_match.group(1), input_match.group(1).strip()
    return None, None

def agent_node(state: MessagesState):
    """支持多轮工具调用的 Agent 节点"""
    user_input = state["messages"][-1].content
    format_instructions = parser.get_format_instructions()
    prompt = SYSTEM_PROMPT.format(format_instructions=format_instructions)
    full_prompt = f"{prompt}\nQuestion: {user_input}\n"
    
    # 模拟迭代，最多5轮
    max_iterations = 5
    scratchpad = ""
    for _ in range(max_iterations):
        current_prompt = f"{prompt}\nQuestion: {user_input}\n{scratchpad}"
        response = llm.invoke(current_prompt)
        content = response.content
        
        # 尝试解析 Action
        action, action_input = parse_action(content)
        if action:
            tool = tool_map.get(action)
            if tool:
                try:
                    # 调用工具
                    if action == "save_tool":
                        # 假设 Action Input 是 JSON 字符串
                        inputs = json.loads(action_input)
                        tool_result = tool.invoke(inputs)
                    else:
                        tool_result = tool.invoke({"query": action_input})
                except Exception as e:
                    tool_result = f"工具调用失败: {e}"
                # 将本次的思考 + 观察追加到 scratchpad
                scratchpad += f"{content}\nObservation: {tool_result}\n"
                continue  # 继续循环
        # 没有 Action 或 Action 处理完，认为得到最终答案
        # 如果最终答案包含 JSON，并且用户可能希望保存，我们在这里再检查是否调用了 save_tool
        # 如果没有，可以手动保存作为后备
        return {"messages": [response]}
    # 达到最大迭代次数，返回最后的消息
    return {"messages": [response]}

# 构建图
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", "__end__")
graph = graph_builder.compile()

query = input("What can I help you research? ")
result = graph.invoke({"messages": [("human", query)]})
final_message = result["messages"][-1].content
print("模型原始输出:", final_message)

# 提取 JSON 部分
json_match = re.search(r"\{.*\}", final_message, re.DOTALL)
if json_match:
    try:
        structured_response = parser.parse(json_match.group())
        print(structured_response)
        
        # 兜底保存：如果模型没有调用 save_tool，手动保存一次
        # 避免重复保存，可以检查文件内容或记录标志，这里简单判断输出中是否有 "save_tool" 调用痕迹
        if "save_tool" not in final_message.lower():
            print("模型未调用 save_tool，手动保存结果...")
            from tools import save_tool
            save_tool.invoke({
                "data": json_match.group(),
                "filename": "research_output.txt"
            })
    except Exception as e:
        print("JSON 解析错误:", e)
else:
    print("未找到 JSON 输出。")