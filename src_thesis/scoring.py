"""
Evaluation metrics module for RAG system assessment.

This module provides metrics for evaluating Retrieval-Augmented Generation systems:
- F1 Score: Token-level overlap between prediction and ground truth
- Exact Match: Binary correctness metric
- Recall@K: Retrieval quality metric

The Evaluator class encapsulates all metric computation logic with
support for batch evaluation and statistical analysis.

Example:
    >>> from src_thesis.scoring import Evaluator
    >>> evaluator = Evaluator()
    >>> f1 = evaluator.f1_score("Paris is the capital", "Paris")
    >>> print(f"{f1:.2f}")
    0.67
"""

import re
import string
from typing import Dict, List, Any, Set
from collections import Counter
from dataclasses import dataclass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for a dataset.
    
    Attributes:
        count: Number of evaluated samples
        f1: Average F1 score (0.0 to 1.0)
        em: Average Exact Match score (0.0 to 1.0)
        recall: Average Recall@K score (0.0 to 1.0)
        latency: Average latency in seconds
        cost: Average cost in USD
        steps: Average number of reasoning steps
        escalation_rate: Percentage of queries escalated (for hybrid agents)
    """
    count: int
    f1: float
    em: float
    recall: float
    latency: float
    cost: float
    steps: float
    escalation_rate: float = 0.0


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class Evaluator:
    """
    Evaluation metrics calculator for RAG systems.
    
    This class provides methods for computing accuracy and retrieval
    metrics commonly used in question-answering benchmarks.
    
    Example:
        >>> evaluator = Evaluator()
        >>> f1 = evaluator.f1_score("The capital is Paris", "Paris")
        >>> em = evaluator.exact_match_score("Paris", "Paris")
        >>> print(f"F1: {f1:.2f}, EM: {em}")
        F1: 0.67, EM: 1.00
    """
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize text for fair comparison.
        
        Applies the following transformations:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Normalize whitespace
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text string
            
        Example:
            >>> Evaluator.normalize_answer("The quick, brown fox!")
            'quick brown fox'
        """
        def remove_articles(text: str) -> str:
            """Remove English articles from text."""
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def remove_punctuation(text: str) -> str:
            """Remove all punctuation characters."""
            return ''.join(ch for ch in text if ch not in string.punctuation)
        
        def normalize_whitespace(text: str) -> str:
            """Collapse multiple spaces into one."""
            return ' '.join(text.split())
        
        # Apply transformations in sequence
        text = str(text).lower()
        text = remove_articles(text)
        text = remove_punctuation(text)
        text = normalize_whitespace(text)
        
        return text
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        F1 is the harmonic mean of precision and recall at the token level.
        This metric is more lenient than exact match, giving partial credit
        for answers that contain the correct information even if phrased differently.
        
        Args:
            prediction: The predicted answer text
            ground_truth: The correct answer text
            
        Returns:
            F1 score between 0.0 (no overlap) and 1.0 (perfect match)
            
        Example:
            >>> evaluator = Evaluator()
            >>> evaluator.f1_score("Paris, France", "Paris")
            0.666...  # 2/3 because "Paris" matches but "France" doesn't
        """
        # Normalize and tokenize both texts
        pred_tokens = self.normalize_answer(prediction).split()
        gold_tokens = self.normalize_answer(ground_truth).split()
        
        # Count overlapping tokens
        common_tokens = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common_tokens.values())
        
        # Handle edge case: no tokens in prediction or ground truth
        if num_common == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(gold_tokens) if gold_tokens else 0.0
        
        # F1 is harmonic mean
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate binary exact match score.
        
        Returns 1.0 if normalized prediction exactly matches normalized
        ground truth, otherwise 0.0. This is a strict metric that requires
        perfect answers.
        
        Args:
            prediction: The predicted answer text
            ground_truth: The correct answer text
            
        Returns:
            1.0 for exact match, 0.0 otherwise
            
        Example:
            >>> evaluator = Evaluator()
            >>> evaluator.exact_match_score("Paris", "paris")
            1.0  # Matches after normalization
            >>> evaluator.exact_match_score("Paris, France", "Paris")
            0.0  # Not an exact match
        """
        normalized_pred = self.normalize_answer(prediction)
        normalized_gold = self.normalize_answer(ground_truth)
        
        return 1.0 if normalized_pred == normalized_gold else 0.0
    
    def extract_titles_from_text(self, text: str) -> Set[str]:
        """
        Extract document titles from agent reasoning traces.
        
        Searches for document titles in two formats:
        1. Tool output format: "Title: <name>"
        2. Evidence citations: "Evidence: ['Doc 1', 'Doc 2']"
        
        Args:
            text: The reasoning chain or output text
            
        Returns:
            Set of extracted document titles
            
        Example:
            >>> evaluator = Evaluator()
            >>> text = "Title: Paris\\nEvidence: ['France']"
            >>> evaluator.extract_titles_from_text(text)
            {'Paris', 'France'}
        """
        if not text:
            return set()
        
        titles = set()
        
        # Pattern 1: Tool output "Title: <name>"
        tool_matches = re.findall(r"Title:\s*(.*?)(?:\\n|\n|$)", text)
        titles.update(match.strip() for match in tool_matches)
        
        # Pattern 2: Evidence list "Evidence: ['Doc 1', 'Doc 2']"
        evidence_blocks = re.findall(
            r"Evidence:\s*\[(.*?)\]",
            text,
            re.DOTALL | re.IGNORECASE
        )
        for block in evidence_blocks:
            # Extract quoted strings
            quoted_items = re.findall(r"['\"](.*?)['\"]", block)
            titles.update(item.strip() for item in quoted_items)
        
        # Filter out empty or very short titles (likely noise)
        return {title for title in titles if len(title) > 1}
    
    def calculate_recall(self, result_entry: Dict[str, Any]) -> float:
        """
        Calculate Recall@K for retrieval quality.
        
        Recall@K measures what fraction of the required documents were
        actually retrieved by the agent. This helps distinguish between
        retrieval failures (couldn't find the docs) and reasoning failures
        (found the docs but couldn't extract the answer).
        
        Args:
            result_entry: Dictionary containing:
                - 'gold_facts': Ground truth supporting facts
                - 'reasoning_chain': Agent's reasoning trace
                - 'predicted_raw': Agent's final output
                
        Returns:
            Recall score between 0.0 (nothing found) and 1.0 (all found)
            
        Example:
            >>> evaluator = Evaluator()
            >>> entry = {
            ...     'gold_facts': {'title': ['Paris', 'France']},
            ...     'reasoning_chain': 'Title: Paris\\n...'
            ... }
            >>> evaluator.calculate_recall(entry)
            0.5  # Found 1 out of 2 required documents
        """
        # Extract ground truth titles
        gold_facts = result_entry.get('gold_facts')
        if not gold_facts:
            return 0.0
        
        gold_titles = set()
        if isinstance(gold_facts, dict):
            gold_titles = set(gold_facts.get('title', []))
        elif isinstance(gold_facts, list):
            # HotpotQA format: list of [title, sentence_id] pairs
            gold_titles = set(item[0] for item in gold_facts if item)
        
        if not gold_titles:
            return 0.0
        
        # Extract retrieved titles from reasoning trace and output
        # Combine both fields to catch citations in different places
        combined_text = (
            str(result_entry.get('reasoning_chain', '')) + "\n" +
            str(result_entry.get('predicted_raw', ''))
        )
        
        retrieved_titles = self.extract_titles_from_text(combined_text)
        
        # Normalize both sets for fair comparison
        normalized_retrieved = {
            self.normalize_answer(title) for title in retrieved_titles
        }
        normalized_gold = {
            self.normalize_answer(title) for title in gold_titles
        }
        
        # Calculate recall
        found = normalized_retrieved.intersection(normalized_gold)
        recall = len(found) / len(normalized_gold)
        
        return recall
    
    def evaluate_run(self, results: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Evaluate an entire experimental run.
        
        Computes aggregate metrics across all results including accuracy,
        efficiency, and retrieval quality metrics.
        
        Args:
            results: List of result dictionaries from agent execution
                Each dict should contain:
                - predicted_raw: Agent's answer
                - gold_answer: Correct answer
                - latency_seconds: Processing time
                - token_usage: API usage statistics
                - (optional) gold_facts: For recall calculation
                
        Returns:
            EvaluationMetrics object with all computed metrics
            
        Example:
            >>> evaluator = Evaluator()
            >>> results = [
            ...     {
            ...         'predicted_raw': 'Answer: Paris',
            ...         'gold_answer': 'Paris',
            ...         'latency_seconds': 2.5,
            ...         'token_usage': {'total_cost_usd': 0.001}
            ...     }
            ... ]
            >>> metrics = evaluator.evaluate_run(results)
            >>> print(f"F1: {metrics.f1:.2f}, Cost: ${metrics.cost:.4f}")
        """
        # Initialize accumulators
        total_f1 = 0.0
        total_em = 0.0
        total_recall = 0.0
        total_latency = 0.0
        total_cost = 0.0
        total_steps = 0
        escalation_count = 0
        
        count = len(results)
        
        for entry in results:
            # Extract and parse answer
            raw_pred = entry.get('predicted_raw', '')
            
            # Try to extract "Answer: ..." format
            match = re.search(
                r"Answer:\s*(.*?)(?:\nEvidence|\n|$)",
                raw_pred,
                re.DOTALL | re.IGNORECASE
            )
            prediction = match.group(1).strip() if match else raw_pred.strip()
            
            gold = entry.get('gold_answer', '')
            
            # Accuracy metrics
            total_f1 += self.f1_score(prediction, gold)
            total_em += self.exact_match_score(prediction, gold)
            total_recall += self.calculate_recall(entry)
            
            # Efficiency metrics
            total_latency += entry.get('latency_seconds', 0)
            
            # Parse token usage
            usage = entry.get('token_usage', {})
            if 'error' not in usage:
                total_cost += usage.get('total_cost_usd', 0)
                
                # Extract step count (varies by agent type)
                steps = self._extract_step_count(usage, entry)
                total_steps += steps
                
                # Track hybrid agent routing
                if usage.get('route') == 'AGENT':
                    escalation_count += 1
        
        # Calculate averages
        n = max(1, count)  # Avoid division by zero
        
        return EvaluationMetrics(
            count=count,
            f1=total_f1 / n,
            em=total_em / n,
            recall=total_recall / n,
            latency=total_latency / n,
            cost=total_cost / n,
            steps=total_steps / n,
            escalation_rate=(escalation_count / n) * 100
        )
    
    def _extract_step_count(
        self,
        usage: Dict[str, Any],
        entry: Dict[str, Any]
    ) -> int:
        """
        Extract step count from usage statistics or reasoning chain.
        
        Different agents store step counts in different ways, so we
        need to check multiple fields.
        
        Args:
            usage: Token usage dictionary
            entry: Full result entry
            
        Returns:
            Number of reasoning steps taken
        """
        # Try explicit step fields first
        if 'critique_loops' in usage:
            return 1 + usage['critique_loops']  # Self-correcting
        elif 'turns' in usage:
            return usage['turns']  # Network
        elif 'steps' in usage:
            return usage['steps']  # ReAct / Hybrid
        
        # Fallback: Count replanning steps
        reasoning_chain = entry.get('reasoning_chain', '')
        if "[REPLANNER]" in reasoning_chain:
            return 1 + reasoning_chain.count("[REPLANNER]")
        
        # Default to 1 step
        return 1


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# =============================================================================

# Global evaluator instance
_default_evaluator = Evaluator()


def normalize_answer(text: str) -> str:
    """Normalize text using default evaluator."""
    return _default_evaluator.normalize_answer(text)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score using default evaluator."""
    return _default_evaluator.f1_score(prediction, ground_truth)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match using default evaluator."""
    return _default_evaluator.exact_match_score(prediction, ground_truth)


def calculate_recall(result_entry: Dict[str, Any]) -> float:
    """Calculate recall using default evaluator."""
    return _default_evaluator.calculate_recall(result_entry)


def evaluate_run(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate a run using default evaluator.
    
    Returns dict instead of EvaluationMetrics for backward compatibility.
    """
    metrics = _default_evaluator.evaluate_run(results)
    
    return {
        "count": metrics.count,
        "f1": metrics.f1,
        "em": metrics.em,
        "recall": metrics.recall,
        "latency": metrics.latency,
        "cost": metrics.cost,
        "steps": metrics.steps,
        "escalation_rate": metrics.escalation_rate
    }


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Evaluator...")
    
    evaluator = Evaluator()
    
    # Test normalization
    assert evaluator.normalize_answer("The Paris!") == "paris"
    print("✓ Normalization works")
    
    # Test F1
    f1 = evaluator.f1_score("Paris, France", "Paris")
    assert 0.6 < f1 < 0.7  # Approximately 2/3
    print(f"✓ F1 score works: {f1:.3f}")
    
    # Test exact match
    em = evaluator.exact_match_score("Paris", "paris")
    assert em == 1.0
    print("✓ Exact match works")
    
    # Test title extraction
    text = "Title: Paris\nEvidence: ['France', 'Europe']"
    titles = evaluator.extract_titles_from_text(text)
    assert "Paris" in titles and "France" in titles
    print(f"✓ Title extraction works: {titles}")
    
    print("\n✅ All tests passed!")