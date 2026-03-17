"""
Experiment orchestration script for running all agent benchmarks.

This module coordinates the execution of all seven agent architectures
in sequence, ensuring consistent experimental conditions.

Key Features:
    - Unique run ID for each experimental session
    - Automatic database restart between runs
    - Worker count configuration per agent
    - Automatic results aggregation and visualization

Pipeline:
    1. Create timestamped run directory
    2. Restart Qdrant (clean slate)
    3. Execute all agents sequentially
    4. Generate comparative reports and visualizations
    
Example:
    >>> from src_thesis.run_all import main
    >>> main()
    🧪 THESIS RUN: run_20260214_143052
    📂 Saving to: logs/run_20260214_143052/
    ...
    ✅ DONE. Check logs/run_20260214_143052/
"""

import os
import time
import subprocess
from datetime import datetime
from typing import Dict, List


# =============================================================================
# CONFIGURATION
# =============================================================================

# Worker counts for each agent (tuned to avoid rate limits)
WORKER_CONFIG: Dict[str, int] = {
    "src_thesis.agent_baseline": 10,
    "src_thesis.agent_hybrid": 10,
    "src_thesis.agent_react": 10,
    "src_thesis.agent_supervisor": 10,
    "src_thesis.agent_network": 10,
    "src_thesis.agent_self_correct": 10,
    "src_thesis.agent_plan_execute": 8  # More conservative for stability
}

# Agents to execute (in order)
AGENTS: List[str] = [
    "src_thesis.agent_baseline",
    "src_thesis.agent_hybrid",
    "src_thesis.agent_react",
    "src_thesis.agent_supervisor",
    "src_thesis.agent_self_correct",
    "src_thesis.agent_plan_execute",
    "src_thesis.agent_network"
]


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main() -> None:
    """
    Execute complete experimental pipeline.
    
    This function:
    1. Creates unique run ID based on timestamp
    2. Restarts Qdrant database for clean state
    3. Runs all agents with appropriate worker counts
    4. Generates comparative evaluation report
    5. Creates visualizations
    
    The entire run takes approximately 2-3 hours depending on
    API latency and worker configuration.
    
    Results are saved to: logs/run_YYYYMMDD_HHMMSS/
    
    Example:
        >>> main()
        🧪 THESIS RUN: run_20260214_143052
        🐳 Restarting Database...
        🚀 Running src_thesis.agent_baseline with MAX_WORKERS=10...
        ...
        ✅ DONE. Check logs/run_20260214_143052/
    """
    # Step 1: Setup unique run ID
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    os.environ["THESIS_RUN_ID"] = run_id
    
    print("=" * 60)
    print(f"🧪 THESIS EXPERIMENTAL RUN: {run_id}")
    print("=" * 60)
    print(f"📂 Results will be saved to: logs/{run_id}/")
    print()
    
    # Step 2: Restart Qdrant for clean slate
    print("🐳 Restarting Qdrant Database...")
    try:
        subprocess.run(
            "docker restart qdrant_thesis",
            shell=True,
            check=False  # Don't fail if container doesn't exist
        )
        time.sleep(5)  # Give database time to start
        print("✅ Database restarted")
    except Exception as e:
        print(f"⚠️ Database restart failed (continuing anyway): {e}")
    
    print()
    
    # Step 3: Run all agents sequentially
    for agent in AGENTS:
        worker_count = WORKER_CONFIG.get(agent, 5)
        
        print("-" * 60)
        print(f"🚀 Running {agent}")
        print(f"   Workers: {worker_count}")
        print("-" * 60)
        
        # Prepare environment with worker count
        env = os.environ.copy()
        env["MAX_WORKERS"] = str(worker_count)
        env["THESIS_RUN_ID"] = run_id  # Ensure run ID is passed
        
        try:
            # Execute agent
            subprocess.run(
                f"python -m {agent}",
                shell=True,
                check=False,  # Don't stop if one agent fails
                env=env
            )
            print(f"✅ {agent} completed")
            
        except Exception as e:
            print(f"❌ {agent} failed: {e}")
            print("   Continuing with next agent...")
        
        print()
    
    # Step 4: Generate evaluation report
    print("=" * 60)
    print("📊 Generating Comparative Evaluation Report...")
    print("=" * 60)
    
    try:
        subprocess.run(
            "python -m src_thesis.evaluate_results",
            shell=True,
            env=os.environ  # Pass run ID to evaluation
        )
        print("✅ Evaluation complete")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    # Step 5: Final summary
    print()
    print("=" * 60)
    print("🎉 EXPERIMENTAL RUN COMPLETE")
    print("=" * 60)
    print(f"📁 All results saved to: logs/{run_id}/")
    print()
    print("Generated files:")
    print(f"   - final_metrics_summary.csv")
    print(f"   - failure_mode_analysis.csv")
    print(f"   - plot_effectiveness.png")
    print(f"   - plot_speed_greed.png")
    print(f"   - plot_step_profile.png")
    print(f"   - plot_variance_boxplot.png")
    print(f"   - plot_threshold_calibration.png")
    print(f"   - plot_efficiency_frontier.png")
    print()
    print(f"📊 View results: ls -lh logs/{run_id}/")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Run complete thesis experiment from command line.
    
    Usage:
        python -m src_thesis.run_all
        
    This will:
    - Create timestamped run directory
    - Restart Qdrant database
    - Run all 7 agents (1000 questions each)
    - Generate comparative reports
    - Create 6 visualizations
    
    Estimated time: 2-3 hours
    Estimated cost: ~$10-15 USD (API calls)
    
    Environment Variables:
        THESIS_DEBUG: Set to '1' for verbose logging
    """
    main()