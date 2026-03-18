"""
Self-Correcting agent implementation with Generate-Critique loop.

This module implements a self-correction pattern where the agent
generates candidate answers and then critiques them before submission.

Architecture:
    Query → Retrieve → Generate → [Reflect → Approve/Reject]ⁿ → Answer

Key Characteristics:
    - Iterative self-improvement loop
    - Catches hallucinations before output
    - High accuracy (57.3% F1) but unstable
    - High accuracy (57.3% F1) but unstable
    - Significant crash rate (6.7%) due to infinite critique loops

Example:
    >>> from src_thesis.agent_self_correct import SelfCorrectAgent
    >>> agent = SelfCorrectAgent(model="gpt-4o-mini")
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
from typing import Literal, Annotated, Dict, List, Any

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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


@tool(description="Submit final answer.")
def final_answer(answer: str, evidence: list[str]) -> None:
    """Submit the final answer with evidence."""
    pass


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State for self-correction graph.
    
    Attributes:
        messages: Conversation history
        critique_count: Number of rejection loops
        proposed_answer: Current answer proposal
    """
    messages: Annotated[list, add_messages]
    critique_count: Annotated[int, lambda x, y: x + y]
    proposed_answer: AIMessage


# =============================================================================
# PROMPTS
# =============================================================================

GENERATOR_PROMPT = """You are a rigorous research assistant.

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on information from 'retrieve_wiki_tool'.
2. Be extremely concise. Your final answer should only contain the direct answer entity.
3. EVIDENCE FORMAT: Your 'evidence' list must contain ONLY the EXACT TITLES of the documents you used. Do NOT paste the content sentences.
   - CORRECT: Evidence: ["Paris", "France (Country)"]
   - WRONG: Evidence: ["Paris is the capital.", "France is a country."]
4. STRICT SEARCH LIMIT: You have a maximum of 3 search attempts.
   - If you have searched 3 times, you MUST call 'final_answer' immediately.
   - If you are unsure after 3 searches, state "I don't know"."""

CRITIC_PROMPT = """You are a strict fact-checker.

Check criteria:
1. Is the answer based ONLY on the evidence?
2. Is the answer extremely concise?
3. FORMAT CHECK: Does the 'evidence' field contain ONLY Document Titles (short strings)? If it contains full sentences, you MUST REJECT it.

Output 'APPROVE' if perfect, otherwise 'REJECT' with feedback."""


# =============================================================================
# AGENT CLASS
# =============================================================================

class SelfCorrectAgent:
    """
    Self-correcting agent with generate-critique loop.
    
    This agent implements iterative refinement by:
    1. Generating a candidate answer
    2. Critiquing it for errors
    3. Refining if rejected
    
    The loop continues until approval or max iterations.
    
    Attributes:
        model: The LLM model to use
        max_critiques: Maximum critique iterations (default: 3)
        recursion_limit: Maximum graph steps (default: 30)
        
    Example:
        >>> agent = SelfCorrectAgent(model="gpt-4o-mini")
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
        max_critiques: int = 3,
        recursion_limit: int = 30
    ):
        """
        Initialize the self-correcting agent.
        
        Args:
            model: OpenAI model identifier
            max_critiques: Maximum critique loops before forcing approval
            recursion_limit: Maximum graph execution steps
        """
        self.model = model
        self.max_critiques = max_critiques
        self.recursion_limit = recursion_limit
    
    def _generator_node(self, state: AgentState) -> Dict[str, List]:
        """
        Generate node that creates candidate answers.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with generated message
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        ).bind_tools([retrieve_wiki_tool, final_answer])
        
        history = [m for m in state["messages"] if m.type != "system"]
        response = llm.invoke([SystemMessage(content=GENERATOR_PROMPT)] + history)
        
        return {"messages": [response]}
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute tool calls and enforce search limits.
        
        This node:
        1. Executes retrieval or answer submission tools
        2. Tracks search count
        3. Forces answer submission after 3 searches
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with tool results and optional proposal
        """
        last_message = state["messages"][-1]
        outputs = []
        proposal = None
        
        # Step 1: Execute tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "retrieve_wiki_tool":
                    result = retrieve_wiki_tool.invoke(tool_call)
                    outputs.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"]
                        )
                    )
                elif tool_call["name"] == "final_answer":
                    proposal = last_message
                    outputs.append(
                        ToolMessage(
                            content="Submitted.",
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"]
                        )
                    )
        
        # Step 2: Enforce search limit (prevent infinite loops)
        search_count = sum(
            1 for m in state["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
            and m.tool_calls[0]["name"] == "retrieve_wiki_tool"
        )
        
        # Force submission after 3 searches
        if search_count >= 3:
            outputs.append(
                HumanMessage(
                    content="[SYSTEM]: MAXIMUM SEARCH STEPS REACHED. You have searched 3 times. "
                            "You MUST call 'final_answer' now with the best info you have."
                )
            )
        
        if proposal:
            return {"messages": outputs, "proposed_answer": proposal}
        else:
            return {"messages": outputs}
    
    def _critic_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Critic node that reviews proposed answers.
        
        Args:
            state: Current graph state with proposed answer
            
        Returns:
            Dict with approval or rejection message
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        class Critique(TypedDict):
            """Critique response structure."""
            decision: Literal["APPROVE", "REJECT"]
            feedback: str
        
        # Extract proposed answer
        args = state["proposed_answer"].tool_calls[0]["args"]
        review = f"Proposed: {args.get('answer')}\nEvidence: {args.get('evidence')}"
        
        history = [m for m in state["messages"] if m.type != "system"]
        
        # Get critique
        result = llm.with_structured_output(Critique).invoke(
            [SystemMessage(content=CRITIC_PROMPT)] + history + [HumanMessage(content=review)]
        )
        
        # Force approval after max critiques to prevent infinite loops
        current_loops = state.get("critique_count", 0)
        if result["decision"] == "APPROVE" or current_loops >= self.max_critiques:
            # Approved - emit final answer
            return {
                "messages": [
                    AIMessage(
                        content=f"Answer: {args.get('answer')}\nEvidence: {args.get('evidence')}"
                    )
                ],
                "critique_count": 0
            }
        
        # Rejected - request refinement
        return {
            "messages": [HumanMessage(content=f"Critique: {result['feedback']}")],
            "critique_count": 1
        }
    
    def _route_tools(self, state: AgentState) -> str:
        """
        Route after tool execution.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        # Check if proposal exists
        if state.get("proposed_answer"):
            return "critic"
        return "generator"
    
    def _route_critic(self, state: AgentState) -> str:
        """
        Route after critic decision.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        last_message = state["messages"][-1]
        
        # If approved (AIMessage), end
        if isinstance(last_message, AIMessage):
            return END
        
        # If too many critiques, force end
        if state.get("critique_count", 0) > self.max_critiques:
            return END
        
        # Otherwise refine
        return "generator"
    
    def build_graph(self) -> StateGraph:
        """
        Construct the self-correction graph.
        
        Graph Structure:
            START → generator → tools → critic → [generator or END]
        
        Returns:
            Compiled LangGraph state machine
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("critic", self._critic_node)
        
        # Define edges
        workflow.add_edge(START, "generator")
        workflow.add_edge("generator", "tools")
        
        workflow.add_conditional_edges(
            "tools",
            self._route_tools,
            {"critic": "critic", "generator": "generator"}
        )
        
        workflow.add_conditional_edges(
            "critic",
            self._route_critic,
            {END: END, "generator": "generator"}
        )
        
        return workflow.compile()
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through self-correction loop.
        
        Args:
            question_data: Dictionary with question data
            
        Returns:
            Dictionary with results
        """
        # Add jitter
        time.sleep(random.uniform(0.1, 1.0))
        
        query = question_data['question']
        start_time = time.time()
        
        # Build graph
        app = self.build_graph()
        
        token_usage = {}
        final_message = ""
        reasoning_chain = ""
        
        try:
            with get_openai_callback() as cb:
                try:
                    final_state = app.invoke(
                        {"messages": [HumanMessage(content=query)], "critique_count": 0},
                        config={"recursion_limit": self.recursion_limit}
                    )
                    final_message = final_state["messages"][-1].content
                    
                except Exception as inner_e:
                    print(f"⚠️ Self-Correct execution error: {inner_e}")
                    final_message = "Error: Recursion limit reached"
                    final_state = {
                        "messages": [HumanMessage(content=f"Error: {inner_e}")],
                        "critique_count": self.max_critiques
                    }
            
            # Build trace
            reasoning_chain = self._build_trace(final_state.get("messages", []))
            
            token_usage = {
                "total_cost_usd": cb.total_cost,
                "total_tokens": cb.total_tokens,
                "critique_loops": final_state.get("critique_count", 0)
            }
            
        except Exception as e:
            final_message = f"Error: {e}"
            reasoning_chain = str(e)
            token_usage = {"error": str(e)}
        
        return {
            "question_id": question_data['id'],
            "question": query,
            "gold_answer": question_data['answer'],
            "gold_facts": question_data['supporting_facts'],
            "predicted_raw": final_message,
            "reasoning_chain": reasoning_chain,
            "latency_seconds": time.time() - start_time,
            "token_usage": token_usage
        }
    
    def _build_trace(self, messages: List) -> str:
        """Build readable trace from messages."""
        trace = []
        for m in messages:
            role = "CRITIC" if "Critique:" in str(m.content) else m.type.upper()
            trace.append(f"[{role}]: {m.content}")
        return "\n\n".join(trace)


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_self_correct_experiment(
    model: str = config.LLM_MODEL,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """Run Self-Correcting experiment."""
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"🪞 Starting Self-Correcting Agent Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Max critiques: 3"
    )
    
    agent = SelfCorrectAgent(model=model)
    
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="Self-Correcting"
    )
    
    output_path = config.LOG_DIR / "self_correct_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """Run Self-Correcting experiment from CLI."""
    model = os.getenv("SELFCORRECT_MODEL", config.LLM_MODEL)
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_self_correct_experiment(model=model, max_workers=max_workers)