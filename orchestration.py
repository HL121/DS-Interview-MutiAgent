from typing import TypedDict, Optional, List, Dict, Any
import os
from pathlib import Path
from scripts.Agent1.skill_analyzer_agent import SkillAnalyzerAgent
from scripts.scope_planner_agent import ScopePlannerAgent
from scripts.Agent2.agentic_retrieval import init_agentic_retriever
from scripts.Agent2.langchain_retrieval import init_retriever
from scripts.Agent3.Planning_Agent import normalize_tasks, run_planning_agent
from scripts.langchain_llm import get_chat_model
from langgraph.graph import StateGraph, END

client = get_chat_model()


def init_pipeline_retriever():
    use_agentic = os.getenv("USE_AGENTIC_RETRIEVAL", "true").lower() not in {"0", "false", "no"}
    if use_agentic:
        return init_agentic_retriever(llm=client)
    return init_retriever()



class AgentState(TypedDict):

    # User Input
    jd: str
    user_desc: str
    days_left: int

    # Agent 1 Output
    extracted: Optional[Dict[str, Any]]
    mapped: Optional[Dict[str, Any]]
    weights: Optional[Dict[str, int]]

    # Agent scoop planner Output
    total_questions: Optional[int]  
    difficulty_distribution: Optional[Dict[str, float]]
    plan: Optional[Dict[str, int]]

    # Agent 2 Output
    retrieve_questions: Optional[Dict[str, list]]

    # Agent 3 Output
    days: Optional[List[List[Dict]]] 
    summaries: Optional[List[Dict]]

def agent1(state:AgentState):
    jd = state.get("jd")
    user_desc = state.get("user_desc")
    agent = SkillAnalyzerAgent(client=client)
    result = agent.run(jd_text=jd, user_desc=user_desc)
    state["weights"] = result["weights"]
    return {"extracted":result["extracted"], "mapped":result["mapped"], "weights":result["weights"]}

def agent_scope(state:AgentState):
    jd = state.get("jd")
    user_desc = state.get("user_desc")
    weights = state.get("weights")
    days_left = state.get("days_left") 
    agent = ScopePlannerAgent(client=client)
    result = agent.run(user_desc=user_desc, jd_text=jd, skill_weights=weights, days_left=days_left)
    state["total_questions"] = result["total_questions"]
    state["difficulty_distribution"] = result["difficulty_distribution"]
    state["plan"] = result["skill_plan"]
    return {"total":result["total_questions"], "diff":result["difficulty_distribution"], "plan":result["skill_plan"]}

def agent2(state:AgentState):
    plan = state.get("plan")
    jd = state.get("jd") or ""
    user_desc = state.get("user_desc") or ""
    retriever = init_pipeline_retriever()
    retrieve_by_skill = {}
    for skill, k in plan.items():
        num_qs = int(k)
        if num_qs<=0:
            continue
        candidate_count = max(num_qs * 3, num_qs + 5)
        questions = retriever.retrieve(
            query=skill,
            topk=candidate_count,
            fetch=max(30, candidate_count * 3),
            type_filter=None,
            skill_filter=skill,
            difficulty_distribution=state.get("difficulty_distribution"),
            jd_text=jd,
            user_desc=user_desc,
        )
        retrieve_by_skill[skill] = questions

    return {"retrieve_questions": retrieve_by_skill}

def agent3(state:AgentState):
    agent2_output = state.get("retrieve_questions")
    days_left = state.get("days_left")
    user_request = f"Create a {days_left} day plan."
    agent2_tasks = []
    for skill, task in agent2_output.items():
        for item in task:
            item["requested_skill"] = skill
            item["requested_quota"] = int((state.get("plan") or {}).get(skill, 0))
        agent2_tasks.extend(task)
    agent3_input_tasks = normalize_tasks(agent2_tasks)
    days, summaries = run_planning_agent(
        tasks=agent3_input_tasks,
        user_request=user_request,
        days_left=days_left,
        skill_plan=state.get("plan"),
        difficulty_distribution=state.get("difficulty_distribution"),
        jd_text=state.get("jd") or "",
        user_desc=state.get("user_desc") or "",
        use_llm=True,
        client=client
        )

    return {"days": days, "summaries": summaries}

workflow = StateGraph(AgentState)
workflow.add_node("agent1", agent1)
workflow.add_node("agent_scope", agent_scope)
workflow.add_node("agent2", agent2)
workflow.add_node("agent3", agent3)

workflow.set_entry_point("agent1")
workflow.add_edge("agent1", "agent_scope") 
workflow.add_edge("agent_scope", "agent2")
workflow.add_edge("agent2", "agent3")
workflow.add_edge("agent3", END)

agent_all = workflow.compile()

def multi_agent(jd,user_desc,days_left):
    initial_state : AgentState = {
        "jd": jd,
        "user_desc": user_desc,
        "days_left": days_left
    }
    final_state = agent_all.invoke(initial_state)
    return {
        "days": final_state.get("days"),
        "summaries": final_state.get("summaries"),
        "total_questions": final_state.get("total_questions"),
        "difficulty_distribution": final_state.get("difficulty_distribution"),
        "skill_plan": final_state.get("plan"),
    }
