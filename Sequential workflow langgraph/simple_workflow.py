from langgraph.graph import StateGraph , START ,END
from typing import TypedDict
from PIL import Image
import io

class bmi_state(TypedDict):
    weight : float 
    height : float 
    bmi    : float 
    catigory : str
    

def bmi_calculator(state : bmi_state  ) -> bmi_state:
    weight = state['weight']
    height = state['height']
    bmi = weight/(height **2)
    state['bmi'] = round(bmi, 2)
    return state

def cat_bmi(state : bmi_state) -> bmi_state:
    bmi = state['bmi']
    if bmi <18.5:
        state['catigory'] = "under weight"
    elif 18.5 <= bmi < 25:
        state['catigory'] = "Normal"
    elif 25 <= bmi <30:
        state["catigory"] = "over weight"
    else: 
        state['catigory'] = "obese"
        
    return state


        


graph = StateGraph(bmi_state)

graph.add_node("bmi_node" , bmi_calculator)
graph.add_node("cat_node" , cat_bmi)


graph.add_edge(START ,"bmi_node")
graph.add_edge("bmi_node" , "cat_node")

graph.add_edge("cat_node" , END)


workflow = graph.compile()


initial_state = {"weight": 80  , "height" : 1.75}

final_state = workflow.invoke(initial_state)


png_data=workflow.get_graph().draw_mermaid_png()

img = Image.open(io.BytesIO(png_data))
img.show()