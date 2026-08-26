from langgraph.graph import StateGraph , START , END
from typing import TypedDict 
 


class batsmanState(TypedDict):
    runs : int 
    balls : int 
    fours : int 
    sixes : int 
    sr : float 
    bpb : float 
    boundary_percent : float
    summary :str
    

def strike(state : batsmanState) -> batsmanState:
    sr = (state['runs']/state['balls']) * 100
    return {"sr": sr}

def boundary_per_ball(state : batsmanState) ->batsmanState:
    bpb =state['balls'] / (state['fours'] + state['sixes'] )
    return {"bpb" : bpb}

def boundary_percent(state : batsmanState )-> batsmanState:
    b_per = ((state['fours']) * 4 + (state['sixes'] * 6) / state['runs'])*100
   
    return {"boundary_percent" : b_per}


def summary(state : batsmanState) -> batsmanState:
    summary = f""" 
    Strike - rate {state['sr']} \n
    Boundary_per_ball {state['bpb']} \n
    Boundary_percent  {state['boundary_percent']}
    
    """    
    return {"summary" : summary}
    

graph = StateGraph(batsmanState)

graph.add_node("strike" , strike)
graph.add_node("boundary_per_ball" , boundary_per_ball)
graph.add_node("boundary_percent" , boundary_percent)
graph.add_node("summary_node" , summary)

graph.add_edge(START , "strike")
graph.add_edge(START , "boundary_per_ball")
graph.add_edge(START , "boundary_percent")

graph.add_edge("strike" , "summary_node")
graph.add_edge("boundary_per_ball" , "summary_node")
graph.add_edge("boundary_percent", "summary_node")
graph.add_edge("summary_node" , END)


workflow = graph.compile()

initial_state = {"runs":100 , "balls":50 , "fours": 6 ,"sixes" : 4}
result = workflow.invoke(initial_state)
print(result)