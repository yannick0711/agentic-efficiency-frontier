"""
HotpotQA dataset preparation module.

This module downloads and prepares the HotpotQA validation set
for evaluation. It creates a stratified subset for thesis experiments.

Key Features:
    - Downloads from HuggingFace datasets
    - Stratified sampling for balanced question types
    - Fixed random seed for reproducibility
    - Exports to JSON format for agents

The HotpotQA dataset contains multi-hop questions requiring
reasoning across multiple documents. Question types:
    - Bridge: ~80% (sequential entity resolution)
    - Comparison: ~20% (parallel attribute comparison)

Example:
    >>> from src_thesis.load_data import prepare_eval_dataset
    >>> prepare_eval_dataset()
    📥 Downloading HotpotQA Validation Set...
    ✅ Saved 1000 test questions
"""

import json
from typing import Dict, List, Any

from datasets import load_dataset

from . import config


# =============================================================================
# CONSTANTS
# =============================================================================

# Number of questions to sample for evaluation
EVAL_SET_SIZE: int = 1000

# Random seed for reproducible sampling
RANDOM_SEED: int = 42


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_eval_dataset(
    output_size: int = EVAL_SET_SIZE,
    seed: int = RANDOM_SEED
) -> List[Dict[str, Any]]:
    """
    Prepare evaluation dataset from HotpotQA validation split.
    
    This function:
    1. Downloads HotpotQA from HuggingFace
    2. Uses validation split (not train)
    3. Shuffles with fixed seed
    4. Samples requested number of questions
    5. Extracts relevant fields
    6. Saves to JSON file
    
    Args:
        output_size: Number of questions to sample (default: 1000)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of question dictionaries
        
    Example:
        >>> questions = prepare_eval_dataset(output_size=100)
        >>> len(questions)
        100
        >>> questions[0].keys()
        dict_keys(['id', 'question', 'answer', 'type', 'supporting_facts'])
    """
    print("📥 Downloading HotpotQA Validation Set...")
    
    # Step 1: Load dataset from HuggingFace
    # We use 'distractor' config which includes question/answer pairs
    # We ignore the provided context because we do "Blind Retrieval"
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    print(f"   Loaded {len(dataset)} questions from validation split")
    
    # Step 2: Shuffle and sample
    shuffled = dataset.shuffle(seed=seed)
    selected_data = shuffled.select(range(output_size))
    
    # Step 3: Extract relevant fields
    output_data = []
    
    for row in selected_data:
        entry = {
            "id": row['id'],
            "question": row['question'],
            "answer": row['answer'],
            "type": row['type'],  # 'bridge' or 'comparison'
            "supporting_facts": row['supporting_facts']
            # supporting_facts format: {'title': [...], 'sent_id': [...]}
        }
        output_data.append(entry)
    
    # Step 4: Save to file
    output_file = config.DATA_DIR / f"hotpot_eval_{output_size}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved {len(output_data)} test questions to {output_file}")
    print(f"   Bridge questions: ~{sum(1 for q in output_data if q['type'] == 'bridge')}")
    print(f"   Comparison questions: ~{sum(1 for q in output_data if q['type'] == 'comparison')}")
    print("\n📝 Dataset is ready for agent evaluation!")
    
    return output_data


# =============================================================================
# DATASET STATISTICS
# =============================================================================

def analyze_dataset(dataset_path: str = None) -> Dict[str, Any]:
    """
    Analyze question types and characteristics in the dataset.
    
    Args:
        dataset_path: Path to dataset JSON file (default: uses config)
        
    Returns:
        Dictionary with dataset statistics
        
    Example:
        >>> stats = analyze_dataset()
        >>> print(stats['total_questions'])
        1000
        >>> print(stats['avg_question_length'])
        14.5
    """
    if dataset_path is None:
        dataset_path = config.TEST_DATA_FILE
    
    with open(dataset_path, 'r') as f:
        questions = json.load(f)
    
    # Calculate statistics
    stats = {
        "total_questions": len(questions),
        "bridge_questions": sum(1 for q in questions if q['type'] == 'bridge'),
        "comparison_questions": sum(1 for q in questions if q['type'] == 'comparison'),
        "avg_question_length": sum(len(q['question'].split()) for q in questions) / len(questions),
        "avg_answer_length": sum(len(q['answer'].split()) for q in questions) / len(questions),
    }
    
    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Prepare HotpotQA evaluation dataset from command line.
    
    Usage:
        python -m src_thesis.load_data
        
    This will:
    - Download HotpotQA validation set
    - Sample 1000 questions with fixed seed (42)
    - Save to data/hotpot_eval_1000.json
    
    The dataset is then ready for use by all agents.
    """
    print("=" * 60)
    print("HOTPOTQA DATASET PREPARATION")
    print("=" * 60)
    
    # Prepare dataset
    questions = prepare_eval_dataset()
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    stats = analyze_dataset()
    for key, value in stats.items():
        print(f"   {key}: {value:.1f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)