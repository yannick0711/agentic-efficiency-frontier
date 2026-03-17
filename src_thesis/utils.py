"""
Utility functions for parallel execution and robust error handling.

This module provides infrastructure for running agent experiments:
- Parallel execution with ThreadPoolExecutor
- Automatic retry logic for transient failures
- Graceful error handling and recovery
- Progress tracking with tqdm

The parallel execution system includes:
- Staggered worker initialization to prevent thundering herd
- Exponential backoff on retries
- Soft error detection in agent outputs

Example:
    >>> from src_thesis.utils import run_parallel_experiment
    >>> results = run_parallel_experiment(
    ...     worker_func=process_question,
    ...     questions=question_list,
    ...     max_workers=10
    ... )
"""

import time
import random
import concurrent.futures
from typing import Callable, List, Dict, Any
from tqdm import tqdm


# =============================================================================
# RETRY LOGIC
# =============================================================================

def run_with_retry(
    func: Callable[[Dict[str, Any]], Dict[str, Any]],
    arg: Dict[str, Any],
    retries: int = 3,
    delay: int = 5
) -> Dict[str, Any]:
    """
    Execute function with automatic retry on soft failures.
    
    This function runs the provided worker function and inspects its
    return value for error indicators. If errors are detected, it
    retries with exponential backoff.
    
    Soft errors detected:
    - Return value starts with "Error:"
    - token_usage dict contains 'error' key
    - Empty or missing output
    
    Args:
        func: Worker function that processes one question
        arg: Dictionary argument to pass to func (question data)
        retries: Maximum number of retry attempts (default: 3)
        delay: Base delay between retries in seconds (default: 5)
            Actual delay uses exponential backoff with jitter
        
    Returns:
        Dictionary with question results, or error dict if all retries fail
        
    Example:
        >>> def process(q):
        ...     return {'predicted_raw': 'Answer: Paris'}
        >>> result = run_with_retry(process, {'question': 'Capital?'})
        >>> result['predicted_raw']
        'Answer: Paris'
    """
    last_result = None
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            # Step 1: Execute the worker function
            result = func(arg)
            last_result = result
            
            # Step 2: Inspect for soft errors in return value
            raw_text = result.get('predicted_raw', '')
            
            # Check for explicit error messages
            if str(raw_text).startswith("Error"):
                raise ValueError(f"Agent returned error text: {raw_text}")
            
            # Check for error flag in usage stats
            usage = result.get('token_usage', {})
            if isinstance(usage, dict) and 'error' in usage:
                raise ValueError(f"Token usage error: {usage['error']}")
            
            # Check for empty output
            if not raw_text.strip():
                raise ValueError("Agent returned empty answer.")
            
            # Step 3: If we reach here, result is valid
            return result
            
        except Exception as e:
            last_error = e
            
            # Only retry if we have attempts left
            if attempt < retries:
                # Exponential backoff with jitter
                # Prevents thundering herd on retry
                sleep_time = (delay * (2 ** attempt)) + random.uniform(1, 3)
                
                if config.DEBUG_MODE:
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {sleep_time:.1f}s...")
                
                time.sleep(sleep_time)
    
    # Step 4: All retries exhausted
    # Return last result if available, otherwise create error entry
    if last_result:
        return last_result
    
    # Fallback: Create error result
    return {
        "question_id": arg.get('id', 'unknown'),
        "question": arg.get('question', ''),
        "gold_answer": arg.get('answer', ''),
        "gold_facts": arg.get('supporting_facts', []),
        "predicted_raw": f"Error: Critical failure after {retries} retries. {last_error}",
        "reasoning_chain": f"System crash: {last_error}",
        "latency_seconds": 0.0,
        "token_usage": {"error": str(last_error)}
    }


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def run_parallel_experiment(
    worker_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    questions: List[Dict[str, Any]],
    max_workers: int = 5,
    desc: str = "Experiment"
) -> List[Dict[str, Any]]:
    """
    Execute experiment in parallel with automatic retry and ramp-up.
    
    This function orchestrates parallel execution with:
    1. Staggered worker initialization to prevent API rate limits
    2. Automatic retry on failures via run_with_retry
    3. Progress tracking with tqdm
    4. Consistent result ordering
    
    The staggered start prevents "thundering herd" where all workers
    hit the API simultaneously, which can trigger rate limit errors.
    
    Args:
        worker_func: Function to process one question
            Should accept Dict[str, Any] and return Dict[str, Any]
        questions: List of question dictionaries to process
        max_workers: Maximum number of concurrent threads (default: 5)
        desc: Description for progress bar (default: "Experiment")
        
    Returns:
        List of result dictionaries, sorted by question_id
        
    Example:
        >>> def process_q(q):
        ...     return {'question_id': q['id'], 'predicted_raw': 'Answer'}
        >>> questions = [{'id': '1', 'question': 'Test?'}]
        >>> results = run_parallel_experiment(process_q, questions, max_workers=2)
        >>> len(results)
        1
    """
    results = []
    
    print(
        f"🚀 Running parallel experiment: {desc}\n"
        f"   Questions: {len(questions)}\n"
        f"   Workers: {max_workers}\n"
        f"   Features: Auto-retry + Staggered start"
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dictionary to track futures
        future_to_question = {}
        
        # Submit tasks with staggered start to prevent thundering herd
        for i, question in enumerate(questions):
            # Submit task wrapped in retry logic
            future = executor.submit(run_with_retry, worker_func, question)
            future_to_question[future] = question
            
            # Stagger initial worker submissions
            # This prevents all workers from hitting API at exactly t=0
            if i < max_workers:
                time.sleep(2.0)  # 2 second delay between initial workers
        
        # Collect results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_question),
            total=len(questions),
            desc=desc
        ):
            try:
                result = future.result()
                results.append(result)
                
            except Exception as e:
                # This should rarely happen since run_with_retry handles errors
                question = future_to_question[future]
                print(f"❌ Critical thread crash for question {question.get('id')}: {e}")
                
                # Create error result
                results.append({
                    "question_id": question.get('id', 'unknown'),
                    "question": question.get('question', ''),
                    "gold_answer": question.get('answer', ''),
                    "gold_facts": question.get('supporting_facts', []),
                    "predicted_raw": f"Error: Thread crash: {e}",
                    "reasoning_chain": f"Critical error: {e}",
                    "latency_seconds": 0.0,
                    "token_usage": {"error": str(e)}
                })
    
    # Sort results by question ID for consistency
    results.sort(key=lambda x: x.get('question_id', ''))
    
    return results


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the parallel execution utilities."""
    print("Testing Parallel Execution Utils...")
    
    # Mock worker function
    def mock_worker(question: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate processing a question."""
        time.sleep(random.uniform(0.1, 0.3))  # Simulate work
        return {
            "question_id": question['id'],
            "question": question['question'],
            "predicted_raw": f"Answer to {question['question']}",
            "latency_seconds": 0.2,
            "token_usage": {"total_tokens": 100}
        }
    
    # Create mock questions
    mock_questions = [
        {"id": f"q{i}", "question": f"Question {i}?"}
        for i in range(5)
    ]
    
    # Test parallel execution
    results = run_parallel_experiment(
        worker_func=mock_worker,
        questions=mock_questions,
        max_workers=2,
        desc="Test"
    )
    
    # Verify results
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    assert all('predicted_raw' in r for r in results), "Missing predicted_raw"
    
    print(f"\n✅ Processed {len(results)} questions successfully!")
    print(f"   First result: {results[0]['predicted_raw']}")


# Import config only if needed (avoid circular imports)
try:
    from . import config
except ImportError:
    # Fallback for standalone testing
    class config:
        DEBUG_MODE = False