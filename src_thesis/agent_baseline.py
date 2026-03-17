"""
Baseline RAG agent implementation.

This module implements the simplest possible RAG architecture:
a single-shot retrieval followed by generation. This serves as
the control group for comparing more complex agentic architectures.

Architecture:
    User Query → Vector Search (k=5) → LLM Generation → Answer

Key Characteristics:
    - No loops or iterations
    - Cheapest and fastest architecture
    - Baseline for performance comparison
    - Also tested with gpt-4o for scale comparison

Example:
    >>> from src_thesis.agent_baseline import BaselineAgent
    >>> agent = BaselineAgent(model="gpt-4o-mini")
    >>> result = agent.process_question({
    ...     'id': '123',
    ...     'question': 'What is the capital of France?',
    ...     'answer': 'Paris',
    ...     'supporting_facts': {...}
    ... })
    >>> print(result['predicted_raw'])
    'Answer: Paris'
"""

import json
import time
import random
import os
from typing import Dict, List, Any
from pathlib import Path

from langchain_community.callbacks.manager import get_openai_callback

from . import config
from .llm_client import call_llm
from .retrieval_tool import search_wiki
from .utils import run_parallel_experiment


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a helpful and rigorous research assistant. 
You will be given a question and a set of retrieved Wikipedia abstracts.

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on the provided Context. Do not use outside knowledge.
2. Be extremely concise. Your final answer should only contain the direct answer entity (e.g., a name, a date, a place). Do NOT explain your reasoning.
3. Your Final Answer MUST be in the specific format below.

Final Answer Format:
Answer: [Your concise answer entity]
Evidence: ["Document Title 1", "Document Title 2"]
"""


# =============================================================================
# AGENT CLASS
# =============================================================================

class BaselineAgent:
    """
    Single-shot RAG baseline agent.
    
    This agent performs one retrieval and one generation step,
    without any loops, self-correction, or multi-agent interaction.
    
    Attributes:
        model: The LLM model to use (e.g., 'gpt-4o-mini')
        k: Number of documents to retrieve (default: 5)
        
    Example:
        >>> agent = BaselineAgent(model="gpt-4o-mini", k=5)
        >>> question_data = {
        ...     'id': '123',
        ...     'question': 'Who invented the telephone?',
        ...     'answer': 'Alexander Graham Bell',
        ...     'supporting_facts': [...]
        ... }
        >>> result = agent.process_question(question_data)
        >>> print(result['predicted_raw'])
    """
    
    def __init__(self, model: str = config.LLM_MODEL, k: int = 5):
        """
        Initialize the baseline agent.
        
        Args:
            model: OpenAI model identifier
            k: Number of documents to retrieve per query
        """
        self.model = model
        self.k = k
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single question through the baseline pipeline.
        
        Pipeline:
            1. Add random jitter to prevent thundering herd
            2. Retrieve top-k documents from vector database
            3. Generate answer using LLM
            4. Log reasoning chain and compute metrics
        
        Args:
            question_data: Dictionary with keys:
                - 'question': The query string
                - 'id': Unique question identifier
                - 'answer': Ground truth answer
                - 'supporting_facts': Gold standard documents
                
        Returns:
            Dictionary with evaluation results:
                - question_id: Original ID
                - question: Query text
                - gold_answer: Correct answer
                - gold_facts: Supporting documents
                - predicted_raw: Agent's output
                - reasoning_chain: Full trace
                - latency_seconds: Processing time
                - token_usage: API usage stats
                
        Example:
            >>> agent = BaselineAgent()
            >>> result = agent.process_question({
            ...     'id': '1',
            ...     'question': 'Capital of France?',
            ...     'answer': 'Paris',
            ...     'supporting_facts': []
            ... })
            >>> result['question_id']
            '1'
        """
        # Add jitter to prevent all workers hitting API simultaneously
        time.sleep(random.uniform(0.1, 1.5))
        
        query = question_data['question']
        start_time = time.time()
        
        # Initialize result containers
        token_usage = {}
        final_answer = ""
        reasoning_chain = ""
        
        try:
            # Step 1: Retrieve relevant documents
            context_text = search_wiki(query, k=self.k)
            
            # Step 2: Generate answer
            user_message = f"Question: {query}\n\nContext:\n{context_text}"
            
            response = call_llm(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.0  # Deterministic for reproducibility
            )
            
            final_answer = response['text']
            
            # Step 3: Build reasoning trace for analysis
            reasoning_chain = (
                f"[HUMAN]: {query}\n\n"
                f"[TOOL (RETRIEVE)]: {context_text}\n\n"
                f"[AI]: {final_answer}"
            )
            
            # Step 4: Calculate costs
            # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
            usage = response['usage']
            cost_usd = (
                (usage.get('prompt_tokens', 0) * 0.15 / 1_000_000) +
                (usage.get('completion_tokens', 0) * 0.60 / 1_000_000)
            )
            
            token_usage = {
                "total_tokens": usage.get('total_tokens', 0),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_cost_usd": cost_usd,
                "successful_requests": 1,
                "steps": 1,  # Baseline always takes 1 step
                "tool_calls": 1  # One retrieval call
            }
        
        except Exception as e:
            # Handle errors gracefully
            final_answer = f"Error: {e}"
            reasoning_chain = f"Error during processing: {e}"
            token_usage = {"error": str(e)}
        
        # Build result dictionary
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

def run_baseline_experiment(
    model: str = config.LLM_MODEL,
    k: int = 5,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Run the baseline experiment on the full evaluation set.
    
    This function:
    1. Loads test questions from disk
    2. Processes them in parallel using ThreadPoolExecutor
    3. Saves results to logs directory
    
    Args:
        model: LLM model to use
        k: Number of documents to retrieve
        max_workers: Number of parallel workers
        
    Returns:
        List of result dictionaries
        
    Example:
        >>> results = run_baseline_experiment(
        ...     model="gpt-4o-mini",
        ...     max_workers=5
        ... )
        >>> print(f"Processed {len(results)} questions")
    """
    # Load evaluation questions
    questions_path = config.DATA_DIR / "hotpot_eval_1000.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    # Get worker count from environment or use default
    worker_count = int(os.getenv("MAX_WORKERS", max_workers))
    
    print(
        f"🧪 Starting Baseline RAG Experiment\n"
        f"   Model: {model}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {worker_count}\n"
        f"   Retrieval: top-{k} documents"
    )
    
    # Create agent instance
    agent = BaselineAgent(model=model, k=k)
    
    # Run in parallel with automatic retry logic
    results = run_parallel_experiment(
        worker_func=agent.process_question,
        questions=questions,
        max_workers=worker_count,
        desc="Baseline"
    )
    
    # Save results
    output_filename = "baseline_results.json"
    if model == "gpt-4o":
        output_filename = "baseline_results_4o.json"
    
    output_path = config.LOG_DIR / output_filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved results to {output_path}")
    
    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Run baseline experiment from command line.
    
    Environment Variables:
        MAX_WORKERS: Number of parallel workers (default: 10)
        BASELINE_MODEL: Override model (default: from config)
        BASELINE_K: Override retrieval count (default: 5)
    """
    # Allow environment variable overrides
    model = os.getenv("BASELINE_MODEL", config.LLM_MODEL)
    k = int(os.getenv("BASELINE_K", "5"))
    max_workers = int(os.getenv("MAX_WORKERS", "10"))
    
    run_baseline_experiment(
        model=model,
        k=k,
        max_workers=max_workers
    )