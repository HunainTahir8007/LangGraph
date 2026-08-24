import sys
sys.stdout.reconfigure(encoding="utf-8")

from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from typing import TypedDict
from PIL import Image
import io

load_dotenv()

llm =  ChatGroq(model ="llama-3.3-70b-versatile" , temperature=0.3 )

class llmState(TypedDict):
    question : str 
    answer   : str


def llm_node(state : llmState) -> llmState:
    question = state["question"]
    prompt = f"After deeply analyzing the question :- {question} provide me the suitable answer"
    answer = llm.invoke(prompt)
    state['answer'] = answer.content
    
    return state
    

graph = StateGraph(llmState)

graph.add_node("LLM" , llm_node)

graph.add_edge(START , "LLM")
graph.add_edge("LLM" , END)

workflow = graph.compile()

question = {"question":"what is the purpose of the research paper attention is all u need, who launch this paper?"}

final = workflow.invoke(question)
# print(final['answer'])

img_png = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(img_png))
img.show()
