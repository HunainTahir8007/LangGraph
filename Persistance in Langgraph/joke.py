from langgraph.graph import StateGraph , START , END 
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from typing import TypedDict 
from PIL import Image
import io 
import uuid

load_dotenv()

model = ChatGroq(model ="llama-3.3-70b-versatile" , temperature=0.2)

class JokeState(TypedDict):
    topic : str 
    joke : str 
    explaination : str 
    

def make_joke(state : JokeState) -> JokeState:
    topic = state['topic']
    prompt = f"give the funniest joke on the topic \n {topic}"
    output = model.invoke(prompt).content
    return {"joke" : output}

def make_explaination(state : JokeState) -> JokeState:
    joke = state['joke']
    prompt = f"Give me the deatailed explaination about this joke \n {joke}"
    output = model.invoke(prompt).content
    return {"explaination" : output}

graph = StateGraph(JokeState)

graph.add_node("joke" , make_joke)
graph.add_node("explain" , make_explaination)
graph.add_edge(START , "joke")
graph.add_edge("joke" , "explain")
graph.add_edge("explain" , END)

cheakpoint = InMemorySaver()
workflow = graph.compile(checkpointer=cheakpoint)
config1 = {"configurable":{"thread_id": "1"}}
inp = {"topic" : "Artificial Intelligence"}
out = workflow.invoke(inp , config=config1)

workflow.get_state(config=config1)
re = list(workflow.get_state_history(config=config1))
print(workflow.get_state({"configurable":{"thread_id": "1" , "checkpoint_id" :"1f186b6a-e3ae-6329-8000-ac7d34320664"}}))
gg= workflow.update_state({"configurable":{"thread_id": "1" , "checkpoint_id" :"1f186b6a-e3ae-6329-8000-ac7d34320664"}}
, {"topic" : "samosa"})

print(list(workflow.get_state_history(config=config1)))