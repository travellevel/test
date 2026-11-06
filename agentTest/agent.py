AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]
3. Action 必须严格写成：Action: 工具名(参数名="参数值")

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 参数值必须用英文双引号包裹，参数赋值必须用等号 "="
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

from ollama import Client

class OllamaDeepSeekClient:
    """
    一个用于调用 Ollama 中 deepseek-r1:7b 模型的客户端。
    """
    def __init__(self, model: str = "deepseek-r1:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = Client(host=host)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 Ollama API 来生成回应。"""
        print("正在调用 Ollama 大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response['message']['content']
            print("Ollama 大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用 Ollama API 时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


