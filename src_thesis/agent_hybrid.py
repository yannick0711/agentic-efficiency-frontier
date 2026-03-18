"""
Hybrid Adaptive Agent implementation with confidence-based routing.

This module implements an adaptive agent that dynamically routes queries
based on confidence scores, balancing cost and accuracy.

Architecture:
    Query → Router (Confidence δ) → [Fast Path or Complex Path] → Answer

Key Characteristics:
    - Best ROI: 55.0% F1 at $0.00052 (31% cheaper than Network)
    - Confidence threshold δ=0.6 (calibrated via sensitivity analysis)
    - Bimodal execution: 59% fast, 41% complex
    - Lowest variance (σ=2.89%) - most stable architecture

Example:
    >>> from src_thesis.agent_hybrid import HybridAgent
    >>> agent = HybridAgent(model="gpt-4o-mini", threshold=0.6)
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
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import config
from .retrieval_tool import search_wiki
from .utils import run_parallel_experiment
from .agent_network import NetworkAgent


# =============================================================================
# MODELS
# =============================================================================

class RouterDecision(BaseModel):
    """
    Router decision with confidence score.
    
    Attributes:
        reasoning: One sentence explaining the decision
        answer: Extracted answer or "I don't know"
        confidence_score: Float from 0.0 (guessing) to 1.0 (certain)
    """
    reasoning: str = Field(
        description="One sentence reasoning. Is the context sufficient?"
    )
    answer: str = Field(
        description="The extracted answer or 'I don't know'"
    )
    confidence_score: float = Field(
        description="0.0 (Guessing) to 1.0 (Certain). If partial info, give 0.4-0.6."
    )


# =============================================================================
# PROMPTS
# =============================================================================

ROUTER_PROMPT = """You are a QA system. Read the Context and answer the Question.

Context: {context}
Question: {question}

INSTRUCTIONS:
1. Answer the question based ONLY on the context.
2. Rate your CONFIDENCE (0.0 to 1.0):
   - 1.0: The exact answer is explicitly stated.
   - 0.8: The answer is strongly implied or requires simple logic.
   - 0.5: I found some relevant info, but the specific entity is missing.
   - 0.1: The context is irrelevant.

Do not be too strict. If the answer is likely correct, give a high score."""


# =============================================================================
# AGENT CLASS
# =============================================================================

class HybridAgent:
    """
    Adaptive router agent with confidence-based escalation.
    
    This agent implements a two-tier architecture:
    1. Fast path: Baseline RAG for high-confidence queries
    2. Slow path: Network debate for low-confidence queries
    
    The routing decision is based on a calibrated confidence threshold.
    
    Attributes:
        model: The LLM model to use
        threshold: Confidence threshold for routing (default: 0.6)
        k: Number of documents to retrieve for baseline (default: 5)
        
    Example:
        >>> agent = HybridAgent(model="gpt-4o-mini", threshold=0.6)
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
        threshold: float = 0.6,
        k: int = 5
    ):
        """
        Initialize the hybrid agent.
        
        Args:
            model: OpenAI model identifier
            threshold: Confidence score threshold for routing
            k: Number of documents for baseline retrieval
        """
        self.model = model
        self.threshold = threshold
        self.k = k
        
        # Initialize Network agent for complex queries
        self.network_agent = NetworkAgent(model=model)
    
    def _run_baseline_router(
        self,
        query: str,
        context_text: str
    ) -> Dict[str, Any]:
        """
        Run baseline RAG with confidence scoring.
        
        This method performs a quick baseline retrieval and generation,
        then asks the model to self-evaluate its confidence.
        
        Args:
            query: User question
            context_text: Retrieved context
            
        Returns:
            Dict with 'text', 'confidence', 'usage'
        """
        llm = ChatOpenAI(
            model=self.model,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0
        )
        
        structured_llm = llm.with_structured_output(RouterDecision)
        
        with get_openai_callback() as cb:
            try:
                result = structured_llm.invoke(
                    ROUTER_PROMPT.format(context=context_text, question=query)
                )
                final_confidence = result.confidence_score
                final_text = result.answer
                
            except Exception as e:
                print(f"⚠️ Router error: {e}")
                return {
                    "text": "Error",
                    "confidence": 0.0,
                    "usage": {
                        "total_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    }
                }
        
        return {
            "text": final_text,
            "confidence": final_confidence,
            "usage": {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }
        }
    
    def _run_expert_agent(self, query: str) -> tuple[str, int]:
        """
        Run the Network (Debate) agent for complex queries.
        
        Args:
            query: User question
            
        Returns:
            Tuple of (answer_text, step_count)
        """
        # Build and execute network graph
        app = self.network_agent.build_graph()
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        # Execute with safety limit
        final_state = app.invoke(
            inputs,
            config={"recursion_limit": 25}
        )
        
        # Extract answer and count steps
        final_message = final_state["messages"][-1].content
        steps = len(final_state["messages"])
        
        return final_message, steps
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question through adaptive routing.
        
        Pipeline:
            1. Add jitter for rate limiting
            2. Baseline retrieval + confidence scoring
            3. Route based on threshold:
               - High confidence (≥ δ) → Return baseline answer
               - Low confidence (< δ) → Escalate to Network agent
            4. Calculate combined metrics
        
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
                - reasoning_chain: Full trace with routing decision
                - latency_seconds: Processing time
                - token_usage: Combined statistics including route info
        """
        # Add jitter
        time.sleep(random.uniform(0.05, 0.2))
        
        query = question_data['question']
        start_time = time.time()
        
        # Initialize metrics
        token_usage = {
            "total_tokens": 0,
            "total_cost_usd": 0,
            "steps": 0,
            "route": "BASELINE"  # Default route
        }
        
        final_answer = ""
        reasoning_chain = ""
        
        try:
            # Step 1: Baseline attempt with confidence scoring
            context_text = search_wiki(query, k=self.k)
            baseline_result = self._run_baseline_router(query, context_text)
            
            # Update costs from baseline
            usage = baseline_result['usage']
            cost = (
                usage.get('prompt_tokens', 0) * 0.15 / 1_000_000 +
                usage.get('completion_tokens', 0) * 0.60 / 1_000_000
            )
            token_usage["total_tokens"] += usage.get('total_tokens', 0)
            token_usage["total_cost_usd"] += cost
            token_usage["steps"] += 1
            
            confidence = baseline_result['confidence']
            
            # Build initial trace
            reasoning_chain += f"[TOOL (RETRIEVE)]: {context_text}\n\n"
            reasoning_chain += (
                f"[BASELINE]: Confidence {confidence:.2f}\n"
                f"Output: {baseline_result['text']}\n\n"
            )
            
            # Step 2: Routing decision
            if confidence >= self.threshold:
                # ✅ High confidence - use baseline answer
                final_answer = baseline_result['text']
                token_usage["route"] = "BASELINE"
                
            else:
                # ⚠️ Low confidence - escalate to Network agent
                reasoning_chain += (
                    f"⚠️ Confidence {confidence:.2f} < {self.threshold}. "
                    f"Switching to Network (Debate)...\n\n"
                )
                token_usage["route"] = "AGENT"
                
                with get_openai_callback() as cb:
                    try:
                        agent_answer, steps = self._run_expert_agent(query)
                        final_answer = agent_answer
                        reasoning_chain += f"[NETWORK (DEBATE)]: {agent_answer}"
                        
                    except Exception as e:
                        print(f"⚠️ Network agent failed: {e}")
                        # Fallback to baseline
                        final_answer = baseline_result['text']
                        reasoning_chain += (
                            f"\n[FALLBACK]: Network failed. "
                            f"Reverting to baseline answer."
                        )
                        steps = 15
                
                # Update costs from network agent
                token_usage["total_tokens"] += cb.total_tokens
                token_usage["total_cost_usd"] += cb.total_cost
                token_usage["steps"] += steps
        
        except Exception as e:
            final_answer = f"Error: {e}"
            reasoning_chain = f"Error during processing: {e}"
            token_usage["error"] = str(e)
        
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


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_hybrid_experiment(
    model: str = config.LLM_MODEL,
    threshold: float = 0.6,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Run Hybrid agent experiment.
    
    Args:
        model: LLM model to use
        threshold: Confidence threshold for routing
        max_workers: Number of parallel workers
        
    Returns:
        List of result dictionaries
    """
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"🔀 Starting Hybrid Adaptive Agent Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Threshold: δ={threshold}\n"
        f"   Expert: Network (Debate)"
    )
    
    agent = HybridAgent(model=model, threshold=threshold)
    
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc=f"Hybrid_T{int(threshold*100)}"
    )
    
    # Save with threshold in filename
    output_filename = f"hybrid_results.json"
    output_path = config.LOG_DIR / output_filename
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Run Hybrid experiment from CLI.
    
    Environment Variables:
        MAX_WORKERS: Number of parallel workers (default: 10)
        HYBRID_MODEL: Override model (default: from config)
        CONFIDENCE_THRESHOLD: Override threshold (default: 0.6)
    """
    model = os.getenv("HYBRID_MODEL", config.LLM_MODEL)
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_hybrid_experiment(
        model=model,
        threshold=threshold,
        max_workers=max_workers
    )