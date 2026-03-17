"""
Failure mode analysis module for error classification.

This module analyzes agent failures and classifies them into:
    - Retrieval Failures: Missing gold documents
    - Reasoning Failures: Gold documents present but answer incorrect
    - Crashes: Recursion limit or control flow errors

This helps identify whether architectural improvements should target
search capabilities or reasoning capabilities.

Example:
    >>> from src_thesis.analyze_failure_modes import analyze_failure_modes
    >>> analyze_failure_modes()
    Architecture          | Total Failures | Retrieval Fail | Reasoning Fail | Crashes
    Baseline (4o-mini)    | 624            | 41.6% (260)    | 20.8% (130)    | 0
    ...
"""

import json
import re
from typing import Dict, List, Any, Tuple

import pandas as pd

from . import config
from .scoring import f1_score


# =============================================================================
# CONFIGURATION
# =============================================================================

# Architecture files to analyze
FILES = [
    ("Baseline (4o-mini)", "baseline_results.json"),
    ("Baseline (4o)",      "baseline_results_4o.json"),
    ("ReAct",             "react_results.json"),
    ("Supervisor",        "supervisor_results.json"),
    ("Plan-Execute",      "plan_execute_results.json"),
    ("Self-Correcting",   "self_correct_results.json"),
    ("Network",           "network_results.json"),
    ("Hybrid (T=0.6)",    "hybrid_results.json")
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def robust_extract_answer(text: str) -> str:
    """
    Extract answer from agent output using same logic as scoring.
    
    Args:
        text: Raw agent output
        
    Returns:
        Extracted answer string
    """
    if not isinstance(text, str):
        return ""
    
    # Use regex to find "Answer:" content
    match = re.search(
        r"Answer:\s*(.*?)(?:\nEvidence|\n|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    
    # Fallback: Clean up common prefixes
    clean = text.replace("**Answer:**", "").replace("Answer:", "")
    if "Evidence:" in clean:
        clean = clean.split("Evidence:")[0]
    
    return clean.strip()


def extract_context_from_chain(chain: str) -> str:
    """
    Extract retrieved content from reasoning chain.
    
    Args:
        chain: Reasoning chain string
        
    Returns:
        Concatenated content from all retrievals
    """
    if not isinstance(chain, str):
        return ""
    
    context_text = ""
    
    # Strategy 1: Look for "Content:" blocks
    content_matches = re.findall(
        r'Content:\s*(.*?)(?=\n---|\[|\n\n)',
        chain,
        re.DOTALL
    )
    if content_matches:
        context_text += " ".join(content_matches)
    
    # Strategy 2: Network agent content='...' format
    network_matches = re.findall(
        r"content=['\"](.*?)['\"] name=",
        chain,
        re.DOTALL
    )
    if network_matches:
        context_text += " ".join(network_matches)
    
    return context_text


# =============================================================================
# CLASSIFICATION LOGIC
# =============================================================================

def classify_failure(
    entry: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Classify a single question result as success or specific failure type.
    
    Args:
        entry: Result dictionary from agent execution
        
    Returns:
        Tuple of (is_failure, failure_type)
            - is_failure: True if incorrect
            - failure_type: 'retrieval', 'reasoning', 'crash', or 'success'
    """
    # Extract answer and gold
    raw_pred = entry.get('prediction', entry.get('predicted_raw', ""))
    pred = robust_extract_answer(raw_pred)
    gold = entry.get('gold_answer', "")
    
    # Check for crash (UPDATED LOGIC)
    if raw_pred is None or (
        isinstance(raw_pred, str) and (
            "limit" in raw_pred.lower() or 
            "error" in raw_pred.lower() or 
            "agent stopped" in raw_pred.lower() or
            "failed to generate" in raw_pred.lower()
        )
    ):
        return True, 'crash'
    
    # Check correctness
    score = f1_score(pred, str(gold))
    if score > 0.8:
        return False, 'success'
    
    # Answer is wrong - classify why
    # Extract context
    retrieved_raw = entry.get('retrieved_docs', [])
    context = ""
    
    if retrieved_raw and isinstance(retrieved_raw, list):
        for doc in retrieved_raw:
            if isinstance(doc, dict):
                context += " " + doc.get('page_content', "")
            elif isinstance(doc, str):
                context += " " + doc
    else:
        chain = entry.get('reasoning_chain', "")
        context = extract_context_from_chain(chain)
    
    # Check if gold answer is in context
    context_lower = context.lower()
    gold_lower = str(gold).lower().strip()
    
    if len(gold_lower) > 3 and gold_lower in context_lower:
        return True, 'reasoning'
    else:
        return True, 'retrieval'


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_failure_modes() -> pd.DataFrame:
    """
    Analyze failure modes across all architectures.
    
    Returns:
        DataFrame with failure statistics
        
    Example:
        >>> df = analyze_failure_modes()
        >>> print(df[['Architecture', 'Total Failures', 'Retrieval Failure %']])
    """
    print(f"{'Architecture':<20} | {'Total Failures':<15} | {'Retrieval Fail':<22} | {'Reasoning Fail':<22} | {'Crashes':<10}")
    print("-" * 100)
    
    summary_data = []
    
    for arch, filename in FILES:
        path = config.LOG_DIR / filename
        if not path.exists():
            continue
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Initialize counters
        retrieval_errors = 0
        reasoning_errors = 0
        crashes = 0
        total_failures = 0
        
        # Classify each result
        for entry in data:
            is_failure, failure_type = classify_failure(entry)
            
            if is_failure:
                total_failures += 1
                if failure_type == 'crash':
                    crashes += 1
                elif failure_type == 'retrieval':
                    retrieval_errors += 1
                elif failure_type == 'reasoning':
                    reasoning_errors += 1
        
        # Calculate percentages
        safe_total = total_failures if total_failures > 0 else 1
        
        p_ret = (retrieval_errors / safe_total) * 100
        p_rea = (reasoning_errors / safe_total) * 100
        
        # Print row
        print(
            f"{arch:<20} | {total_failures:<15} | "
            f"{p_ret:5.1f}% ({retrieval_errors})     | "
            f"{p_rea:5.1f}% ({reasoning_errors})     | "
            f"{crashes:<10}"
        )
        
        # Store for DataFrame
        summary_data.append({
            "Architecture": arch,
            "Total Failures": total_failures,
            "Retrieval Failure %": p_ret,
            "Retrieval Failure Count": retrieval_errors,
            "Reasoning Failure %": p_rea,
            "Reasoning Failure Count": reasoning_errors,
            "Crashes": crashes
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_path = config.LOG_DIR / "failure_mode_analysis.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved analysis to {output_path}")
    
    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Run failure mode analysis from command line.
    
    Usage:
        python -m src_thesis.analyze_failure_modes
        
    Output:
        - Console table with failure statistics
        - CSV file: logs/*/failure_mode_analysis.csv
    """
    analyze_failure_modes()