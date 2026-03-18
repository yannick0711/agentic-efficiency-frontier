"""
Network (Multi-Agent Debate) agent implementation.

This module implements a multi-agent debate system where a Proposer
suggests answers and a Critic challenges them. A Judge makes the
final decision after structured debate rounds.

Architecture:
    Query → Proposer → Critic → Proposer → Critic → Judge → Answer

Key Characteristics:
    - Multi-agent adversarial debate (4 turns)
    - Reduces sycophancy through role separation
    - Highest accuracy (58.4% F1) but expensive
    - Highest accuracy (58.4% F1) but expensive
    - Low crash rate (2.0%)

Example:
    >>> from src_thesis.agent_network import NetworkAgent
    >>> agent = NetworkAgent(model="gpt-4o-mini")
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
    """
    Search Wikipedia for relevant documents.
    
    Args:
        query: Search query string
        
    Returns:
        Formatted search results with top 3 documents
    """
    return search_wiki(query, k=3)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class NetworkState(TypedDict):
    """
    State for multi-agent debate graph.
    
    Attributes:
        messages: Conversation history with agents
        turn_count: Number of debate turns completed
    """
    messages: Annotated[list, add_messages]
    turn_count: Annotated[int, lambda x, y: x + y]


# =============================================================================
# PROMPTS
# =============================================================================

PROPOSER_PROMPT = """You are Agent A (Proposer), a rigorous research assistant.

Goal: Find the answer to the user's question using tools.

INSTRUCTIONS:
1. Use 'retrieve_wiki_tool' to find evidence.
2. Be direct and concise. Base claims ONLY on retrieved text."""

CRITIC_PROMPT = """You are Agent B (Critic), a rigorous fact-checker.

Goal: Verify the Proposer's claims.

INSTRUCTIONS:
1. Use 'retrieve_wiki_tool' to double-check facts.
2. Be skeptical. Point out missing evidence or hallucinations.
3. If the Proposer is correct, confirm it."""

JUDGE_PROMPT = """You are the Judge.

CRITICAL INSTRUCTIONS:
1. Extract the final answer based ONLY on the evidence in the history.
2. Be extremely concise. Your 'Answer:' must contain ONLY the direct answer entity.
3. Cite ONLY documents from history.

Final Answer Format:
Answer: [Your concise answer entity]
Evidence: ["Document Title 1", "Document Title 2"]"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class NetworkAgent:
    """
    Multi-agent debate system with Proposer, Critic, and Judge.
    
    This agent implements adversarial debate to reduce hallucination
    and improve answer quality. The separation of roles prevents
    sycophancy where the model agrees with its own mistakes.
    
    Attributes:
        model: The LLM model to use
        max_turns: Maximum debate turns (default: 4)
        recursion_limit: Maximum graph steps (default: 30)
        
    Example:
        >>> agent = NetworkAgent(model="gpt-4o-mini")
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
        max_turns: int = 4,
        recursion_limit: int = 30
    ):
        """
        Initialize the Network agent.
        
        Args:
            model: OpenAI model identifier
            max_turns: Number of debate rounds before judge
            recursion_limit: Maximum graph execution steps
        """
        self.model = model
        self.max_turns = max_turns
        self.recursion_limit = recursion_limit
    
    def _run_agent(
        self,
        state: NetworkState,
        prompt: str,
        name: str
    ) -> Dict[str, List]:
        """
        Execute one agent (Proposer or Critic) with tool access.
        
        Args:
            state: Current graph state
            prompt: System prompt for this agent
            name: Agent name for tracking
            
        Returns:
            Dict with updated messages
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        ).bind_tools([retrieve_wiki_tool])
        
        # Get message history (exclude system messages)
        history = [m for m in state["messages"] if m.type != "system"]
        
        # Generate response
        response = llm.invoke([SystemMessage(content=prompt)] + history)
        response.name = name
        
        return {"messages": [response]}
    
    def _tool_node(self, state: NetworkState) -> Dict[str, List]:
        """
        Execute tool calls from the last agent message.
        
        Args:
            state: Current graph state
            
        Returns:
            Dict with tool result messages
        """
        last_message = state["messages"][-1]
        outputs = []
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
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
    
    def _judge_node(self, state: NetworkState) -> Dict[str, List]:
        """
        Final judge node that synthesizes debate into answer.
        
        Args:
            state: Current graph state with full debate history
            
        Returns:
            Dict with judge's final answer
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        history = [m for m in state["messages"] if m.type != "system"]
        response = llm.invoke([SystemMessage(content=JUDGE_PROMPT)] + history)
        response.name = "judge"
        
        return {"messages": [response]}
    
    def _router(self, state: NetworkState) -> str:
        """
        Determine next node based on current state.
        
        Logic:
            1. If last message has tool calls → go to tools
            2. If turn count >= max_turns → go to judge
            3. If last was proposer → go to critic
            4. Otherwise → go to proposer
        
        Args:
            state: Current graph state
            
        Returns:
            Name of next node to execute
        """
        last_message = state["messages"][-1]
        
        # Check for tool calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        
        # Check if debate is complete
        if state.get("turn_count", 0) >= self.max_turns:
            return "judge"
        
        # Alternate between agents
        if last_message.name == "proposer":
            return "critic"
        
        return "proposer"
    
    def _tool_router(self, state: NetworkState) -> str:
        """
        After tools execute, route back to the agent that called them.
        
        Args:
            state: Current graph state
            
        Returns:
            Name of agent to return to
        """
        messages = state["messages"]
        
        # Find the last AI message (before tool results)
        for message in reversed(messages[:-1]):
            if isinstance(message, AIMessage):
                return message.name
        
        # Default to proposer
        return "proposer"
    
    def build_graph(self) -> StateGraph:
        """
        Construct the multi-agent debate graph.
        
        Graph Structure:
            START → proposer → critic → proposer → critic → judge → END
            (with tool nodes between agents as needed)
        
        Returns:
            Compiled LangGraph state machine
        """
        workflow = StateGraph(NetworkState)
        
        # Add agent nodes
        workflow.add_node(
            "proposer",
            lambda s: {
                **self._run_agent(s, PROPOSER_PROMPT, "proposer"),
                "turn_count": 1
            }
        )
        
        workflow.add_node(
            "critic",
            lambda s: {
                **self._run_agent(s, CRITIC_PROMPT, "critic"),
                "turn_count": 1
            }
        )
        
        # Add tool and judge nodes
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("judge", self._judge_node)
        
        # Define edges
        workflow.add_edge(START, "proposer")
        
        workflow.add_conditional_edges(
            "proposer",
            self._router,
            {"tools": "tools", "critic": "critic", "judge": "judge"}
        )
        
        workflow.add_conditional_edges(
            "critic",
            self._router,
            {"tools": "tools", "proposer": "proposer", "judge": "judge"}
        )
        
        workflow.add_conditional_edges(
            "tools",
            self._tool_router,
            {"proposer": "proposer", "critic": "critic"}
        )
        
        workflow.add_edge("judge", END)
        
        return workflow.compile()
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through multi-agent debate.
        
        Pipeline:
            1. Add jitter for rate limiting
            2. Build debate graph
            3. Execute debate rounds
            4. Extract judge's answer
            5. Build reasoning trace
            6. Calculate metrics
        
        Args:
            question_data: Dictionary with:
                - 'question': Query text
                - 'id': Unique identifier
                - 'answer': Ground truth
                - 'supporting_facts': Gold documents
                
        Returns:
            Dictionary with:
                - question_id: Original ID
                - predicted_raw: Final answer
                - reasoning_chain: Full debate transcript
                - latency_seconds: Processing time
                - token_usage: API statistics
        """
        # Add jitter to prevent thundering herd
        time.sleep(random.uniform(0.1, 1.0))
        
        query = question_data['question']
        start_time = time.time()
        
        # Build graph
        app = self.build_graph()
        
        # Initialize result containers
        token_usage = {}
        final_answer = ""
        reasoning_chain = ""
        
        try:
            with get_openai_callback() as cb:
                try:
                    # Execute debate
                    final_state = app.invoke(
                        {
                            "messages": [HumanMessage(content=query)],
                            "turn_count": 0
                        },
                        config={"recursion_limit": self.recursion_limit}
                    )
                    
                    # Extract final answer from judge
                    final_answer = final_state["messages"][-1].content
                    
                except Exception as inner_e:
                    print(f"⚠️ Network execution error: {inner_e}")
                    final_answer = "Error: Network loop limit reached"
                    final_state = {
                        "messages": [AIMessage(content="Error")],
                        "turn_count": self.max_turns
                    }
            
            # Build reasoning trace
            reasoning_chain = self._build_trace(final_state["messages"])
            
            # Calculate metrics
            token_usage = {
                "total_cost_usd": cb.total_cost,
                "total_tokens": cb.total_tokens,
                "turns": final_state.get("turn_count", 0)
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
        """
        Build readable debate transcript from messages.
        
        Args:
            messages: List of messages from graph execution
            
        Returns:
            Formatted multi-agent conversation
        """
        trace = []
        
        for message in messages:
            agent_name = message.name.upper() if message.name else message.type.upper()
            trace.append(f"[{agent_name}]: {message.content}")
        
        return "\n\n".join(trace)


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_network_experiment(
    model: str = config.LLM_MODEL,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Run Network agent experiment on full evaluation set.
    
    Args:
        model: LLM model to use
        max_workers: Number of parallel workers
        
    Returns:
        List of result dictionaries
    """
    # Load evaluation questions
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"🗣️ Starting Network (Multi-Agent Debate) Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Max turns: 4"
    )
    
    # Create agent
    agent = NetworkAgent(model=model)
    
    # Run experiment
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="Network"
    )
    
    # Save results
    output_path = config.LOG_DIR / "network_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Run Network experiment from command line.
    
    Environment Variables:
        MAX_WORKERS: Number of parallel workers (default: 10)
        NETWORK_MODEL: Override model (default: from config)
    """
    model = os.getenv("NETWORK_MODEL", config.LLM_MODEL)
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_network_experiment(model=model, max_workers=max_workers)