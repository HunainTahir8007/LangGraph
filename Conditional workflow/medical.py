from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph , START ,END 
from pydantic import BaseModel , Field 
from typing  import TypedDict , Annotated , Literal 
from PIL import Image 
import io 


load_dotenv()
model = ChatGroq(model ="llama-3.3-70b-versatile" , temperature=0.3)

class Structuredoutput(BaseModel):
    risk : Literal["low" , "medium" , "high"] = Field(description="Severity level based on reported physical symptoms.")
    
    
structured_model = model.with_structured_output(Structuredoutput)

class PatientState(TypedDict):
    symptoms : str 
    risk_level : str 
    action_plan : str
def risk_cheaking(state : PatientState ) ->PatientState:
    symptom = state['symptoms']
    prompt = f"After deeply anlyzing the symptoms of patient provide me risk level  low , medium , high \n {symptom}"
    output = structured_model.invoke(prompt)
    return {"risk_level" : output.risk }

def low_risk(state : PatientState) -> PatientState:
    symptoms = state['symptoms']
    prompt = f"You are a nurse. your duty is to analyze the symptoms \n{symptoms} Provide gentle self-care advice and over-the-counter care tips."
    output = model.invoke(prompt).content
    return {'action_plan' : output}

def medium_risk(state : PatientState) -> PatientState:
    symptoms = state['symptoms']
    prompt = f"You are a Senior Doctor. your duty is to analyze the symptoms \n{symptoms} Provide appointement i can take to being normal ."
    output = model.invoke(prompt).content
    return {'action_plan' : output}
def high_risk(state : PatientState) -> PatientState:
    symptoms = state['symptoms']
    prompt = f"You are a Senior Doctor . your duty is to analyze the symptoms \n{symptoms} provide me the emergency advice for high risk"
    output = model.invoke(prompt).content
    return {'action_plan' : output}
def route_risk(state: PatientState) -> str:
    return state["risk_level"]
graph = StateGraph(PatientState)

graph.add_node("risk_cheak" , risk_cheaking)
graph.add_node("low" , low_risk)
graph.add_node("medium" , medium_risk)
graph.add_node("high"  , high_risk)

graph.add_edge(START , "risk_cheak")
graph.add_conditional_edges("risk_cheak" , route_risk , {"low": "low", "medium": "medium", "high": "high"})
graph.add_edge("low", END)
graph.add_edge("medium", END)
graph.add_edge("high", END)
workflow = graph.compile()
png = workflow.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png))
img.show()
in_state = {"symptoms" : "Sudden numbness on the right side of my face, slurred speech, and loss of balance."}
output= workflow.invoke(in_state)
print(output['risk_level'])
print(output['action_plan'])
