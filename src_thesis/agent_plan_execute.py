"""
Plan-and-Execute agent implementation.

This module implements a static planning approach where the agent
generates a complete plan upfront, executes steps sequentially,
and replans if necessary.

Architecture:
    Query → Planner → [Execute Step → Replan]ⁿ → Answer

Key Characteristics:
    - Decoupled planning and execution
    - Sequential step completion
    - Lowest F1 (39.7%) due to open-loop brittleness
    - Cannot recover from flawed initial plans

Example:
    >>> from src_thesis.agent_plan_execute import PlanExecuteAgent
    >>> agent = PlanExecuteAgent(model="gpt-4o-mini")
    >>> result = agent.process_question({
    ...     'id': '123',
    ...     'question': 'Who invented the telephone?',
    ...     'answer': 'Alexander Graham Bell',
    ...     'supporting_facts': []
    ... })
"""

import json
import time
import random
import os
from typing import List, Annotated, Dict, Any

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from . import config
from .retrieval_tool import search_wiki
from .utils import run_parallel_experiment


# =============================================================================
# TOOLS
# =============================================================================

@tool(description="Search Wikipedia abstracts.")
def retrieve_wiki_tool(query: str) -> str:
    """Search Wikipedia for documents."""
    return search_wiki(query, k=3)


# =============================================================================
# MODELS
# =============================================================================

class Plan(BaseModel):
    """Plan with list of steps."""
    steps: List[str] = Field(description="Numbered list of steps")


class Response(BaseModel):
    """Final response with answer and evidence."""
    answer: str = Field(description="Concise answer entity")
    evidence: List[str] = Field(description="Document titles")


class ReplannerOutput(BaseModel):
    """Replanner decision output."""
    new_plan: Plan | None = None
    final_response: Response | None = None


# =============================================================================
# STATE DEFINITION
# =============================================================================

class PlanExecuteState(TypedDict):
    """
    State for plan-execute graph.
    
    Attributes:
        messages: Conversation history
        plan: List of remaining steps
        task: Original user query
        final_response: Final answer if complete
    """
    messages: Annotated[list, add_messages]
    plan: List[str]
    task: str
    final_response: Response | None


# =============================================================================
# PROMPTS
# =============================================================================

PLANNER_PROMPT = """You are an expert researcher. Generate a step-by-step plan to answer the user's question. Be specific."""

EXECUTOR_PROMPT = """You are a rigorous research assistant.

Your goal is to execute the given task using the provided tools.

INSTRUCTIONS:
1. Use specific search queries based on the task.
2. Do not use outside knowledge.
3. Summarize your findings based ONLY on the tool output."""

REPLANNER_PROMPT = """You are a research manager.

DECISION CRITERIA:
1. FINISH: If the history contains the answer.
   - 'answer' must be extremely concise entity.
   - 'evidence' must cite documents from history.
2. UPDATE PLAN: If info missing.
   - **REMOVE the step that was just completed.**
   - Add new steps if necessary.

Original Task: {task}
Current Plan: {plan}"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class PlanExecuteAgent:
    """
    Plan-and-Execute agent with static planning.
    
    This agent generates an upfront plan and executes it sequentially,
    with the ability to replan based on intermediate results.
    
    Attributes:
        model: The LLM model to use
        recursion_limit: Maximum graph steps (default: 15)
        
    Example:
        >>> agent = PlanExecuteAgent(model="gpt-4o-mini")
        >>> result = agent.process_question({
        ...     'id': '1',
        ...     'question': 'Capital of France?',
        ...     'answer': 'Paris',
        ...     'supporting_facts': []
        ... })
    """
    
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        recursion_limit: int = 15
    ):
        """
        Initialize the plan-execute agent.
        
        Args:
            model: OpenAI model identifier
            recursion_limit: Maximum graph execution steps
        """
        self.model = model
        self.recursion_limit = recursion_limit
    
    def _planner_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """
        Planner node that generates initial plan.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with plan and planner message
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        plan = llm.with_structured_output(Plan).invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=state["task"])
        ])
        
        return {
            "plan": plan.steps,
            "messages": [AIMessage(content=f"Plan: {plan.steps}", name="planner")]
        }
    
    def _executor_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """
        Executor node that completes one step from plan.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with executor messages and updated plan
        """
        # Safety check for empty plan
        if not state["plan"]:
            return {
                "messages": [AIMessage(content="Error: Plan is empty", name="executor")]
            }
        
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        ).bind_tools([retrieve_wiki_tool])
        
        # Get current task
        task = state["plan"][0]
        context = [m for m in state["messages"] if m.type != "system"]
        
        # Execute task
        response = llm.invoke([
            SystemMessage(content=EXECUTOR_PROMPT + f"\nTASK: {task}")
        ] + context)
        
        messages = [response]
        
        # Execute any tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = retrieve_wiki_tool.invoke(tool_call)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
        
        # Remove completed step from plan
        remaining_plan = state["plan"][1:]
        
        return {"messages": messages, "plan": remaining_plan}
    
    def _replanner_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """
        Replanner node that decides to finish or continue.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with final response or updated plan
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        history = [m for m in state["messages"] if m.type != "system"]
        current_plan = state.get("plan", [])
        
        response = llm.with_structured_output(ReplannerOutput).invoke([
            SystemMessage(content=REPLANNER_PROMPT.format(
                task=state['task'],
                plan=current_plan
            ))
        ] + history)
        
        # Check if task is complete
        if response.final_response:
            return {
                "final_response": response.final_response,
                "messages": [AIMessage(content="Task Complete", name="replanner")]
            }
        
        # Update plan if needed
        new_steps = response.new_plan.steps if response.new_plan else []
        return {
            "plan": new_steps,
            "messages": [AIMessage(content="Plan Updated", name="replanner")]
        }
    
    def _router_logic(self, state: PlanExecuteState) -> str:
        """
        Determine next node after replanner.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        # If final response exists, end
        if state.get("final_response"):
            return END
        
        # If plan is empty, end (graceful failure)
        if not state.get("plan"):
            return END
        
        # Otherwise continue executing
        return "executor"
    
    def build_graph(self) -> StateGraph:
        """
        Construct the plan-execute graph.
        
        Graph Structure:
            START → planner → executor → replanner → [executor or END]
        
        Returns:
            Compiled LangGraph state machine
        """
        workflow = StateGraph(PlanExecuteState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("replanner", self._replanner_node)
        
        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replanner")
        
        workflow.add_conditional_edges(
            "replanner",
            self._router_logic
        )
        
        return workflow.compile()
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through plan-execute pipeline.
        
        Args:
            question_data: Dictionary with question data
            
        Returns:
            Dictionary with results
        """
        # Add throttle to reduce rate limit errors
        time.sleep(random.uniform(1.0, 2.0))
        
        query = question_data['question']
        start_time = time.time()
        
        # Build graph
        app = self.build_graph()
        
        token_usage = {}
        final_answer = ""
        reasoning_chain = ""
        
        try:
            with get_openai_callback() as cb:
                try:
                    state = app.invoke(
                        {
                            "messages": [],
                            "task": query,
                            "plan": [],
                            "final_response": None
                        },
                        config={"recursion_limit": self.recursion_limit}
                    )
                    
                    # Extract final answer
                    if state.get("final_response"):
                        resp = state["final_response"]
                        final_answer = f"Answer: {resp.answer}\nEvidence: {resp.evidence}"
                    else:
                        final_answer = "Error: Failed to generate final answer."
                        
                except Exception as inner_e:
                    print(f"⚠️ PlanExecute execution error: {inner_e}")
                    final_answer = "Error: Execution limit reached"
                    state = {"final_response": None, "messages": []}
            
            # Build trace
            reasoning_chain = self._build_trace(state.get("messages", []))
            
            token_usage = {
                "total_cost_usd": cb.total_cost,
                "total_tokens": cb.total_tokens
            }
            
        except Exception as e:
            final_answer = f"Error: {e}"
            reasoning_chain = str(e)
            token_usage = {"error": str(e)}
        
        return {
            "question_id": question_data['id'],
            "question": query,
            "gold_answer": question_data['answer'],
            "gold_facts": question_data['supporting_facts'],
            "predicted_raw": final_answer,
            "reasoning_chain": reasoning_chain,
            "latency_seconds": time.time() - start_time,
            "token_usage": token_usage
        }
    
    def _build_trace(self, messages: List) -> str:
        """Build readable trace."""
        trace = []
        for m in messages:
            role = m.name if m.name else m.type
            trace.append(f"[{role.upper()}]: {m.content}")
        return "\n\n".join(trace)


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_plan_execute_experiment(
    model: str = config.LLM_MODEL,
    max_workers: int = 8
) -> List[Dict[str, Any]]:
    """Run Plan-Execute experiment."""
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"📋 Starting Plan-and-Execute Agent Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Max steps: 15"
    )
    
    agent = PlanExecuteAgent(model=model)
    
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="PlanExecute"
    )
    
    output_path = config.LOG_DIR / "plan_execute_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """Run Plan-Execute experiment from CLI."""
    model = os.getenv("PLANEXEC_MODEL", config.LLM_MODEL)
    max_workers = int(os.getenv("MAX_WORKERS", "8"))
    
    run_plan_execute_experiment(model=model, max_workers=max_workers)