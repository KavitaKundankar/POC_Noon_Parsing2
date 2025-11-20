from typing import Optional, List, Dict
from pydantic import BaseModel

class MappingState(BaseModel):
    tenant: str
    parsed_key: str
    standard_key: Optional[str] = None
    candidates: Optional[List[str]] = None
    confidence: Optional[float] = None
    source: Optional[str] = None



async def check_memori(state: MappingState):
    mapping = memori.get(state.tenant, state.parsed_key)
    if mapping:
        state.standard_key = mapping.standard_key
        state.confidence = mapping.confidence
        state.source = "memori"
    return state

async def vector_search(state: MappingState):
    if state.standard_key:
        return state   # memori hit → skip

    results = vectordb.similarity_search(state.tenant, state.parsed_key)
    state.candidates = [doc['metadata']['name'] for doc in results]
    return state

async def llm_chooser(state: MappingState):
    if state.standard_key:
        return state

    sel = await llm_select(state.parsed_key, state.candidates)
    state.standard_key = sel["best_match"]
    state.confidence = sel["confidence"]
    state.source = sel.get("source", "llm")
    return state

async def save_memori(state: MappingState):
    if state.source == "memori":
        return state
    
    if state.confidence >= 0.75:
        memori.save(
            tenant=state.tenant,
            parsed_key=state.parsed_key,
            standard_key=state.standard_key,
            confidence=state.confidence,
        )
    return state

from langgraph.graph import StateGraph, END

workflow = StateGraph(MappingState)

workflow.add_node("check_memori", check_memori)
workflow.add_node("vector_search", vector_search)
workflow.add_node("llm_chooser", llm_chooser)
workflow.add_node("save_memori", save_memori)

workflow.set_entry_point("check_memori")

workflow.add_edge("check_memori", "vector_search")
workflow.add_edge("vector_search", "llm_chooser")
workflow.add_edge("llm_chooser", "save_memori")
workflow.add_edge("save_memori", END)

app = workflow.compile()
