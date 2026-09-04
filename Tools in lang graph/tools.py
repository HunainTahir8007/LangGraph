from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

wrapper = DuckDuckGoSearchAPIWrapper( max_results=5 ) 
search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, subt, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subt":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_model(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition, ["tools", END])
graph.add_edge("tools", "chat_node")

workflow = graph.compile()


out = workflow.invoke({"messages": [HumanMessage(content="tell me about war between USA and iran")]})
print(out["messages"][-1].content)