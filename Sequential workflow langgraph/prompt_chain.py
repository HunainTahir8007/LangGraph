from langgraph.graph import StateGraph , START , END 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from PIL import Image
import io
import sys

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile" , temperature=0.3 )



class blog(TypedDict):
   topic :str 
   outline : str 
   blog : str 

def outline(state : blog ) -> blog:
    topic = state['topic']
    prompt = f"generate me the detailed outline for the topic :- {topic}"
    outline = llm.invoke(prompt).content
    state['outline']= outline
    return state

def explaination(state : blog ) -> blog:
    topic = state['topic']
    outline = state['outline']
    prompt = f"Write the deatailed blog on the topic: {topic} with the given  outline \n {outline}"
    result  = llm.invoke(prompt).content
    state['blog'] = result
    return state

graph = StateGraph(blog)

graph.add_node("outline_node" , outline)
graph.add_node("explaination_node" , explaination)

graph.add_edge(START , "outline_node")
graph.add_edge("outline_node" , "explaination_node")
graph.add_edge("explaination_node" , END)

workflow = graph.compile()

img_png = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(img_png))
# img.show()
topic = {"topic" : "Evolution of AI in India?"}
result = workflow.invoke(topic)
print(result['topic'])
print("*"*50)
print(result['outline'])
print("*"*50)
print(result['blog'])
