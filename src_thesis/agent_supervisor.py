"""
Supervisor (Hierarchical Delegation) agent implementation.

This module implements a hierarchical architecture where a supervisor
manages worker agents, delegating retrieval tasks and synthesizing results.

Architecture:
    Query → Supervisor → [Worker ↔ Supervisor]ⁿ → Synthesis → Answer

Key Characteristics:
    - Hub-and-spoke topology
    - Reduces context pollution via role separation
    - Supervisor routes, workers retrieve
    - High variance (σ=4.06%) due to router flicker

Example:
    >>> from src_thesis.agent_supervisor import SupervisorAgent
    >>> agent = SupervisorAgent(model="gpt-4o-mini")
    >>> result = agent.process_question({
    ...     'id': '123',
    ...     'question': 'Who invented the telephone?',
    ...     'answer': 'Alexander Graham Bell',
    ...     'supporting_facts': []
    ... })
"""

import json
import time
import os
from typing import Literal, Annotated, Dict, List, Any

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
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
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State for supervisor graph.
    
    Attributes:
        messages: Conversation history
        next_step: Next node to execute
    """
    messages: Annotated[list, add_messages]
    next_step: str


# =============================================================================
# PROMPTS
# =============================================================================

SUPERVISOR_PROMPT = """You are a research supervisor.

CRITICAL RULES:
1. You do NOT have access to external knowledge. Use only conversation history.
2. If answer is missing, route to 'researcher'.
3. If answer is present, output 'FINISH'.
4. Reasoning must explain your routing decision."""

RESEARCHER_PROMPT = """You are a rigorous research specialist.

Your goal is to find information to answer the user's question.

INSTRUCTIONS:
1. Generate precise, targeted search queries based on the conversation history.
2. Do NOT answer the question yourself; just gather facts.
3. Use the 'retrieve_wiki_tool' to find evidence."""

SYNTHESIS_PROMPT = """You are a rigorous fact-checker.

CRITICAL INSTRUCTIONS:
1. Base answer ONLY on history.
2. Be extremely concise. Answer entity only. No reasoning.
3. Format: Answer: [Entity]\nEvidence: ["Doc1"]"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class SupervisorAgent:
    """
    Hierarchical delegation agent with supervisor and workers.
    
    This agent separates high-level planning (supervisor) from
    low-level execution (workers), reducing context pollution.
    
    Attributes:
        model: The LLM model to use
        recursion_limit: Maximum graph steps (default: 25)
        
    Example:
        >>> agent = SupervisorAgent(model="gpt-4o-mini")
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
        recursion_limit: int = 25
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            model: OpenAI model identifier
            recursion_limit: Maximum graph execution steps
        """
        self.model = model
        self.recursion_limit = recursion_limit
    
    def _supervisor_node(self, state: AgentState) -> Dict[str, str]:
        """
        Supervisor node that routes to researcher or finish.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with next_step decision
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        class Router(TypedDict):
            """Router response structure."""
            next: Literal["researcher", "FINISH"]
            reasoning: str
        
        response = llm.with_structured_output(Router).invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
        )
        
        return {"next_step": response["next"]}
    
    def _researcher_node(self, state: AgentState) -> Dict[str, List]:
        """
        Worker node that performs retrieval.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with researcher's message
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        ).bind_tools([retrieve_wiki_tool])
        
        response = llm.invoke(
            [SystemMessage(content=RESEARCHER_PROMPT)] + state["messages"]
        )
        response.name = "researcher"
        
        return {"messages": [response]}
    
    def _tool_node(self, state: AgentState) -> Dict[str, List]:
        """
        Execute tool calls from researcher.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with tool results
        """
        last_message = state["messages"][-1]
        outputs = []
        
        if last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                result = retrieve_wiki_tool.invoke(tool_call)
                outputs.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
        
        return {"messages": outputs}
    
    def build_graph(self) -> StateGraph:
        """
        Construct the supervisor graph.
        
        Graph Structure:
            START → supervisor → [researcher → tools → supervisor] or FINISH
        
        Returns:
            Compiled LangGraph state machine
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("tools", self._tool_node)
        
        # Define edges
        workflow.add_edge(START, "supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next_step"],
            {"researcher": "researcher", "FINISH": END}
        )
        
        workflow.add_edge("researcher", "tools")
        workflow.add_edge("tools", "supervisor")
        
        return workflow.compile()
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through supervisor delegation.
        
        Args:
            question_data: Dictionary with question data
            
        Returns:
            Dictionary with results
        """
        query = question_data['question']
        start_time = time.time()
        
        token_usage = {}
        final_answer = ""
        reasoning_chain = ""
        
        try:
            # Build graph
            app = self.build_graph()
            
            # Create synthesizer for final answer
            synthesizer = ChatOpenAI(
                model=self.model,
                api_key=config.OPENAI_API_KEY,
                temperature=0.0
            )
            
            with get_openai_callback() as cb:
                try:
                    # Execute supervision
                    final_state = app.invoke(
                        {"messages": [HumanMessage(content=query)]},
                        config={"recursion_limit": self.recursion_limit}
                    )
                    
                    # Synthesize final answer
                    history = final_state["messages"]
                    final_answer = synthesizer.invoke(
                        [SystemMessage(content=SYNTHESIS_PROMPT)] + history
                    ).content
                    
                except Exception as inner_e:
                    print(f"⚠️ Supervisor execution error: {inner_e}")
                    final_answer = "Error: Supervisor loop limit reached"
                    final_state = {"messages": []}
            
            # Count steps and tools
            steps, tool_calls = self._count_metrics(final_state.get("messages", []))
            
            # Build trace
            reasoning_chain = self._build_trace(final_state.get("messages", []))
            
            token_usage = {
                "total_tokens": cb.total_tokens,
                "total_cost_usd": cb.total_cost,
                "successful_requests": cb.successful_requests,
                "steps": steps + 1,  # +1 for synthesis
                "tool_calls": tool_calls
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
    
    def _count_metrics(self, messages: List) -> tuple[int, int]:
        """Count steps and tool calls."""
        steps = 0
        tool_calls = 0
        
        for message in messages[1:]:  # Skip initial user message
            steps += 1
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_calls += len(message.tool_calls)
        
        return steps, tool_calls
    
    def _build_trace(self, messages: List) -> str:
        """Build readable trace."""
        trace = []
        for m in messages:
            name = m.name.upper() if m.name else m.type.upper()
            trace.append(f"[{name}]: {m.content}")
        return "\n\n".join(trace)


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_supervisor_experiment(
    model: str = config.LLM_MODEL,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """Run Supervisor experiment."""
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"👔 Starting Supervisor Agent Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Max steps: 25"
    )
    
    agent = SupervisorAgent(model=model)
    
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="Supervisor"
    )
    
    output_path = config.LOG_DIR / "supervisor_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """Run Supervisor experiment from CLI."""
    model = os.getenv("SUPERVISOR_MODEL", config.LLM_MODEL)
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_supervisor_experiment(model=model, max_workers=max_workers)