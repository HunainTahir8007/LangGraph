import uuid
from typing import Annotated, TypedDict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt , Command


load_dotenv()

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True,
)
def chat_node(state: ChatbotState) -> ChatbotState:
    decision = interrupt({
        "type":"Human approval needed", 
        "reason" : "can model answer or not ",
        "message" : state['messages'][-1].content,
        "instruction" : "Approve this question or not"
        
    })
    if decision['approved']=="no":
        return {"messages" : AIMessage(content="Not approved")}
    else:
        response = model.invoke(state["messages"]).content
        return {"messages" : [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatbotState)

graph.add_node("Model_talk", chat_node)
graph.add_edge(START, "Model_talk")
graph.add_edge("Model_talk", END)



workflow = graph.compile(checkpointer=checkpointer)
config={"configurable" : {"thread_id" : "1"}}

res=workflow.invoke({"messages" : [
    HumanMessage(content="Explain me the gradient decent in very simple term")
]} , config=config)

user_input = input("Approve or not ")

final_ans = workflow.invoke(
    Command(resume={"approved": user_input}),
    config=config
)
print(final_ans)