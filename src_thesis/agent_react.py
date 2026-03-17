"""
ReAct (Reasoning + Acting) agent implementation.

This module implements the ReAct pattern where the agent alternates
between reasoning about what to do next and taking actions (tool calls).

Architecture:
    Query → [Reason → Act → Observe]ⁿ → Final Answer

Key Characteristics:
    - Iterative reasoning loop
    - Self-directed tool usage
    - Adaptive retrieval depth
    - Maximum 15 steps to prevent infinite loops

Example:
    >>> from src_thesis.agent_react import ReactAgent
    >>> agent = ReactAgent(model="gpt-4o-mini")
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
from typing import Dict, List, Any

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from . import config
from .retrieval_tool import search_wiki
from .utils import run_parallel_experiment


# =============================================================================
# TOOLS
# =============================================================================

@tool
def retrieve_wiki_tool(query: str) -> str:
    """
    Search Wikipedia abstracts for a query.
    
    Args:
        query: The search query string
        
    Returns:
        Formatted search results with top 3 documents
    """
    return search_wiki(query, k=3)


# =============================================================================
# PROMPTS
# =============================================================================

REACT_SYSTEM_PROMPT = """You are a rigorous research assistant.

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on information from 'retrieve_wiki_tool'.
2. Be extremely concise. Your final answer should only contain the direct answer entity.
3. Your Final Answer MUST be in the specific format below.

Final Answer Format:
Answer: [Your concise answer entity]
Evidence: ["Document Title 1", "Document Title 2"]"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class ReactAgent:
    """
    ReAct (Reasoning + Acting) iterative agent.
    
    This agent uses the ReAct pattern to alternate between:
    1. Reasoning about what information is needed
    2. Acting by calling retrieval tools
    3. Observing the results
    
    The loop continues until the agent has enough information or
    reaches the recursion limit.
    
    Attributes:
        model: The LLM model to use
        recursion_limit: Maximum reasoning steps (default: 15)
        
    Example:
        >>> agent = ReactAgent(model="gpt-4o-mini")
        >>> result = agent.process_question({
        ...     'id': '1',
        ...     'question': 'Capital of France?',
        ...     'answer': 'Paris',
        ...     'supporting_facts': []
        ... })
        >>> 'Paris' in result['predicted_raw']
        True
    """
    
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        recursion_limit: int = 15
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            model: OpenAI model identifier
            recursion_limit: Maximum reasoning steps before timeout
        """
        self.model = model
        self.recursion_limit = recursion_limit
    
    def build_agent(self):
        """
        Construct the ReAct agent graph.
        
        Returns:
            Compiled LangGraph agent with ReAct loop
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0  # Deterministic for reproducibility
        )
        
        return create_react_agent(llm, tools=[retrieve_wiki_tool])
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through ReAct reasoning loop.
        
        Pipeline:
            1. Add jitter to prevent API thundering herd
            2. Initialize agent with system prompt
            3. Execute ReAct loop (max 15 steps)
            4. Extract final answer
            5. Build reasoning trace
            6. Calculate metrics
        
        Args:
            question_data: Dictionary containing:
                - 'question': Query text
                - 'id': Unique identifier
                - 'answer': Ground truth answer
                - 'supporting_facts': Gold standard documents
                
        Returns:
            Dictionary with:
                - question_id: Original ID
                - predicted_raw: Agent's answer
                - reasoning_chain: Full trace
                - latency_seconds: Processing time
                - token_usage: API statistics
                
        Example:
            >>> agent = ReactAgent()
            >>> result = agent.process_question({
            ...     'id': '1',
            ...     'question': 'Who invented the telephone?',
            ...     'answer': 'Alexander Graham Bell',
            ...     'supporting_facts': []
            ... })
        """
        # Add jitter to prevent thundering herd
        time.sleep(random.uniform(0.1, 1.0))
        
        query = question_data['question']
        start_time = time.time()
        
        # Build agent for this query
        agent = self.build_agent()
        
        # Initialize result containers
        token_usage = {}
        final_message = ""
        reasoning_chain = ""
        
        try:
            # Prepare input messages
            inputs = {
                "messages": [
                    SystemMessage(content=REACT_SYSTEM_PROMPT),
                    HumanMessage(content=query)
                ]
            }
            
            # Execute agent with usage tracking
            with get_openai_callback() as cb:
                try:
                    response = agent.invoke(
                        inputs,
                        config={"recursion_limit": self.recursion_limit}
                    )
                    final_message = response["messages"][-1].content
                    
                except Exception as inner_e:
                    # Handle recursion limit or other execution errors
                    print(f"⚠️ ReAct execution error: {inner_e}")
                    final_message = "Error: Recursion limit reached"
                    # Create fake response to prevent crashes
                    response = {"messages": [AIMessage(content="Error")]}
            
            # Build reasoning trace from message history
            reasoning_chain = self._build_reasoning_trace(response["messages"])
            
            # Calculate metrics
            steps, tool_calls = self._count_steps_and_tools(response["messages"])
            
            token_usage = {
                "total_tokens": cb.total_tokens,
                "total_cost_usd": cb.total_cost,
                "successful_requests": cb.successful_requests,
                "steps": steps,
                "tool_calls": tool_calls
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
    
    def _build_reasoning_trace(self, messages: List) -> str:
        """
        Build human-readable reasoning trace from message history.
        
        Args:
            messages: List of messages from agent execution
            
        Returns:
            Formatted string showing reasoning steps
        """
        chain = []
        
        # Skip system message (index 0)
        for message in messages[1:]:
            content = message.content
            
            # Annotate tool calls
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_names = ", ".join([tc['name'] for tc in message.tool_calls])
                content += f" [Tool Call: {tool_names}]"
            
            # Format with message type
            chain.append(f"[{message.type.upper()}]: {content}")
        
        return "\n\n".join(chain)
    
    def _count_steps_and_tools(self, messages: List) -> tuple[int, int]:
        """
        Count reasoning steps and tool calls from messages.
        
        Args:
            messages: List of messages from execution
            
        Returns:
            Tuple of (steps, tool_calls)
        """
        steps = 0
        tool_calls = 0
        
        # Skip system + user message (first 2)
        for message in messages[2:]:
            steps += 1
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_calls += len(message.tool_calls)
        
        return steps, tool_calls


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_react_experiment(
    model: str = config.LLM_MODEL,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Run ReAct experiment on full evaluation set.
    
    Args:
        model: LLM model to use
        max_workers: Number of parallel workers
        
    Returns:
        List of result dictionaries
        
    Example:
        >>> results = run_react_experiment(max_workers=5)
        >>> len(results)
        1000
    """
    # Load evaluation questions
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    # Get worker count from environment or use default
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"🤖 Starting ReAct Agent Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Max steps: 15"
    )
    
    # Create agent instance
    agent = ReactAgent(model=model)
    
    # Run in parallel
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="ReAct"
    )
    
    # Save results
    output_path = config.LOG_DIR / "react_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Run ReAct experiment from command line.
    
    Environment Variables:
        MAX_WORKERS: Number of parallel workers (default: 10)
        REACT_MODEL: Override model (default: from config)
    """
    model = os.getenv("REACT_MODEL", config.LLM_MODEL)
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_react_experiment(model=model, max_workers=max_workers)