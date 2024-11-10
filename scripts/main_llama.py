from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

llm = NVIDIA("meta/llama-3.1-70b-instruct", api_key="")

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 20+(2/4)? Calculate step by step ")

print(response)