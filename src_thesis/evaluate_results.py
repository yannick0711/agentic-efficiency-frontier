"""
Results evaluation and visualization module.

This module generates comprehensive comparative analysis and visualizations
for all agent architectures tested in the thesis.

Key Features:
    - Aggregate metrics computation (F1, EM, Recall, Cost, Latency)
    - Statistical significance testing (Wilcoxon signed-rank)
    - Six publication-quality visualizations
    - CSV export for further analysis

Generated Visualizations:
    1. Effectiveness Hierarchy: F1 and EM comparison
    2. Speed & Greed: Cost vs Latency scatter plot
    3. Reasoning Depth Profile: Step distribution stacked bars
    4. Stability Analysis: Variance boxplots with significance
    5. Threshold Calibration: Hybrid sensitivity curve
    6. Efficiency Frontier: Pareto-optimal boundary

Example:
    >>> from src_thesis.evaluate_results import main
    >>> main()
    📊 Evaluating Results in logs/run_20260214/...
    [Generates all plots and metrics]
    ✅ All plots saved to logs/run_20260214/
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from . import config
from .scoring import evaluate_run


# =============================================================================
# PLOT STYLE CONFIGURATION
# =============================================================================

# Match LaTeX document font (Computer Modern) and set high resolution
plt.rcParams.update({
    "text.usetex": False,              # Set True if LaTeX is installed
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,                 # High resolution for all figures
    "savefig.dpi": 300,                # High resolution when saving
    "savefig.bbox": "tight",
})

# =============================================================================
# CONFIGURATION
# =============================================================================

# Architectures to evaluate (in display order)
ARCHITECTURES = [
    ("Baseline (4o-mini)", "baseline_results.json"),
    ("Baseline (4o)", "baseline_results_4o.json"),
    ("ReAct", "react_results.json"),
    ("Supervisor", "supervisor_results.json"),
    ("Self-Correcting", "self_correct_results.json"),
    ("Plan-Execute", "plan_execute_results.json"),
    ("Network", "network_results.json"),
    # Hybrid variants for threshold analysis
    ("Hybrid (T=0.6)", "hybrid_results.json"),
    ("Hybrid Test (T=0.5)", "hybrid_results_test_50.json"),
    ("Hybrid Test (T=0.6)", "hybrid_results_test_60.json"),
    ("Hybrid Test (T=0.7)", "hybrid_results_test_70.json"),
    ("Hybrid Test (T=0.8)", "hybrid_results_test_80.json"),
    ("Hybrid Test (T=0.9)", "hybrid_results_test_90.json")
]

# Main hybrid version for comparison
MAIN_HYBRID_LABEL = "Hybrid (T=0.6)"

# Recursion limits for each architecture
LIMITS = {
    "ReAct": 15,
    "Plan-Execute": 15,
    "Supervisor": 25,
    "Network": 30,
    "Self-Correcting": 30,
    MAIN_HYBRID_LABEL: 25
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_smart_labels(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str
) -> None:
    """
    Add smart positioned labels to avoid overlap.
    
    Args:
        ax: Matplotlib axes object
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        label_col: Column name for labels
    """
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    x_mid = x_min + (x_range / 2)
    offset = x_range * 0.02
    
    for i, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        label = row[label_col]
        
        # Position labels based on x-coordinate
        if x > x_mid:
            ha = 'right'
            final_x = x - offset
        else:
            ha = 'left'
            final_x = x + offset
        
        ax.text(
            final_x, y, label,
            horizontalalignment=ha,
            verticalalignment='center',
            fontsize=9, weight='bold', alpha=0.8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )


def analyze_entry(entry: Dict[str, Any], limit: int) -> Tuple[int, bool]:
    """
    Determine steps taken and whether execution crashed.
    """
    pred = entry.get('predicted_raw', entry.get('prediction'))
    
    # Check for explicit crash indicators
    if pred is None:
        return limit, True
    
    if isinstance(pred, str):
        pred_lower = pred.lower()
        if (
            "limit" in pred_lower or 
            "error" in pred_lower or 
            "agent stopped" in pred_lower or
            "failed to generate" in pred_lower
        ):
            return limit, True
    
    # Extract actual step count
    steps = 1
    usage = entry.get('token_usage', {})
    
    if 'critique_loops' in usage:
        loops = usage['critique_loops']
        steps = 2 + (loops * 2)
    elif 'turns' in usage:
        steps = usage['turns']
    elif 'steps' in usage:
        steps = usage['steps']
    elif "[REPLANNER]" in entry.get('reasoning_chain', ''):
        steps = 1 + entry['reasoning_chain'].count("[REPLANNER]")
    
    # Check if hit the limit (with updated catch-all logic)
    if steps >= limit and (pred is None or (isinstance(pred, str) and (
        "limit" in pred.lower() or 
        "error" in pred.lower() or 
        "agent stopped" in pred.lower() or
        "failed to generate" in pred.lower()
    ))):
        return steps, True
    
    return steps, False


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_step_stacked_bar(
    all_results_map: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
) -> None:
    """
    Generate stacked bar chart showing step distribution.
    
    Args:
        all_results_map: Dict mapping architecture names to result lists
        output_dir: Directory to save plot
    """
    print("   -> Generating Step Distribution Stacked Bar...")
    
    target_agents = [
        "Baseline (4o-mini)",
        "Baseline (4o)",
        "Plan-Execute",
        "Self-Correcting",
        MAIN_HYBRID_LABEL,
        "ReAct",
        "Supervisor",
        "Network",
    ]
    
    data = []
    
    for arch_name, results in all_results_map.items():
        if arch_name not in target_agents:
            continue
        
        limit = LIMITS.get(arch_name, 15)
        
        for entry in results:
            steps, is_crash = analyze_entry(entry, limit)
            
            if is_crash:
                category = "Timed Out (Limit)"
            elif steps == 1:
                category = "Instant (1 Step)"
            elif steps <= 5:
                category = "Short Loop (2-5)"
            elif steps <= 10:
                category = "Medium Loop (6-10)"
            else:
                category = "Deep Loop (11+)"
            
            data.append({"Architecture": arch_name, "Category": category})
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Calculate percentages
    df_counts = df.groupby(['Architecture', 'Category']).size().reset_index(name='Count')
    df_totals = df.groupby('Architecture').size().reset_index(name='Total')
    df_merge = pd.merge(df_counts, df_totals, on='Architecture')
    df_merge['Percent'] = (df_merge['Count'] / df_merge['Total']) * 100
    
    df_pivot = df_merge.pivot(index='Architecture', columns='Category', values='Percent').fillna(0)
    
    # Sort by instant answers
    if "Instant (1 Step)" in df_pivot.columns:
        df_pivot = df_pivot.sort_values("Instant (1 Step)", ascending=False)
    
    # Define color scheme
    category_order = [
        "Instant (1 Step)", "Short Loop (2-5)", "Medium Loop (6-10)",
        "Deep Loop (11+)", "Timed Out (Limit)"
    ]
    existing_cols = [c for c in category_order if c in df_pivot.columns]
    df_pivot = df_pivot[existing_cols]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = {
        "Instant (1 Step)": "#2ca02c",
        "Short Loop (2-5)": "#17becf",
        "Medium Loop (6-10)": "#9467bd",
        "Deep Loop (11+)": "#ff7f0e",
        "Timed Out (Limit)": "#d62728"
    }
    plot_colors = [colors[c] for c in existing_cols]
    
    df_pivot.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, width=0.8)
    
    # Add percentage labels
    for container in ax.containers:
        labels = [f'{w:.0f}%' if w > 5 else '' for w in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center',
                    fontsize=8, color='white', weight='bold')
    
    plt.title("Reasoning Depth Profile: Step Distribution by Architecture")
    plt.ylabel("Percentage of Queries (%)")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Reasoning Depth")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_step_profile.png")
    plt.close()


def generate_speed_greed_plot(df_clean: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate cost vs latency scatter plot.
    
    Args:
        df_clean: DataFrame with metrics
        output_dir: Directory to save plot
    """
    print("   -> Generating Speed & Greed Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(
        data=df_clean, x="Latency (s)", y="Cost ($)",
        hue="Architecture", style="Architecture", s=200,
        palette="deep", ax=ax, legend=False
    )
    
    add_smart_labels(ax, df_clean, "Latency (s)", "Cost ($)", "Architecture")
    
    plt.title("Latency and Cost Trade-off Analysis")
    plt.xlabel("Avg. Latency per Query (s)")
    plt.ylabel("Avg. Cost per Query ($)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_speed_greed.png")
    plt.close()


def generate_effectiveness_plot(df_clean: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate F1 and EM comparison bar chart.
    
    Args:
        df_clean: DataFrame with metrics
        output_dir: Directory to save plot
    """
    print("   -> Generating Effectiveness Hierarchy Plot...")
    
    plt.figure(figsize=(12, 6))
    
    df_melt = df_clean.melt(
        id_vars="Architecture", value_vars=["F1", "EM"],
        var_name="Metric", value_name="Score (%)"
    )
    order = df_clean.sort_values("F1", ascending=False)["Architecture"]
    
    ax = sns.barplot(
        data=df_melt, x="Architecture", y="Score (%)",
        hue="Metric", palette="viridis", order=order
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
    
    plt.title("Effectiveness Hierarchy: Accuracy (F1) & Exact Match (EM)")
    plt.ylabel("Score (%)")
    plt.xlabel("Architecture Strategy")
    plt.ylim(0, 115)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc='upper right', title="Metric")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_effectiveness.png")
    plt.close()


def generate_variance_plot(
    all_results_map: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
) -> None:
    """
    Generate variance boxplot with statistical significance.
    
    Args:
        all_results_map: Dict mapping architecture names to result lists
        output_dir: Directory to save plot
    """
    print("   -> Generating Variance Boxplot with Significance...")
    
    variance_data = []
    BASELINE_ARCH = "Baseline (4o-mini)"
    
    allowed_archs = [
        BASELINE_ARCH, "Baseline (4o)", "ReAct",
        "Supervisor", "Self-Correcting", "Plan-Execute",
        "Network", MAIN_HYBRID_LABEL
    ]
    
    arch_folds = {a: [] for a in allowed_archs}
    
    for arch_name, results in all_results_map.items():
        if arch_name not in allowed_archs:
            continue
        
        sorted_results = sorted(results, key=lambda x: x.get('question_id', str(x.get('_id', ''))))
        total = len(sorted_results)
        if total < 100:
            continue
        
        chunk_size = total // 10
        for i in range(10):
            chunk = sorted_results[i*chunk_size : (i+1)*chunk_size]
            if not chunk:
                continue
            metrics = evaluate_run(chunk)
            score = metrics['f1'] * 100
            
            variance_data.append({
                "Architecture": arch_name,
                "F1 Score": score,
                "Fold": i
            })
            arch_folds[arch_name].append(score)
    
    if not variance_data:
        return
    
    df_var = pd.DataFrame(variance_data)
    
    plt.figure(figsize=(14, 8))
    
    # Sort by median
    order = df_var.groupby("Architecture")["F1 Score"].median().sort_values(ascending=False).index
    
    ax = sns.boxplot(
        data=df_var, x="Architecture", y="F1 Score",
        palette="viridis", order=order, showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
    )
    
    # Statistical significance annotations
    baseline_scores = arch_folds.get(BASELINE_ARCH, [])
    
    if baseline_scores and len(baseline_scores) == 10:
        for i, arch in enumerate(order):
            if arch == BASELINE_ARCH:
                continue
            
            target_scores = arch_folds.get(arch, [])
            if len(target_scores) != 10:
                continue
            
            try:
                stat, p_val = stats.wilcoxon(target_scores, baseline_scores, alternative='greater')
            except ValueError:
                p_val = 1.0
            
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            y_max = df_var[df_var["Architecture"] == arch]["F1 Score"].max()
            color = 'darkgreen' if p_val < 0.05 else 'gray'
            weight = 'bold' if p_val < 0.05 else 'normal'
            
            ax.text(
                i, y_max + 0.5, sig,
                ha='center', va='bottom', color=color, weight=weight, fontsize=12
            )
    
    plt.title("Stability Analysis: Performance Variance & Statistical Significance (vs. Baseline)",
             fontsize=14, pad=15)
    plt.ylabel("F1 Score (%) (across 10 stratified folds)")
    plt.xlabel("")
    
    plt.text(0.02, 0.02,
            f"Significance vs {BASELINE_ARCH}:\n* p<0.05, ** p<0.01, *** p<0.001\nns: not significant (Paired Wilcoxon)",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, style='italic',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_variance_boxplot.png")
    plt.close()


def generate_threshold_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate threshold sensitivity curve for Hybrid agent.
    
    Args:
        df: DataFrame with all metrics including hybrid variants
        output_dir: Directory to save plot
    """
    try:
        hybrid_df = df[df["Architecture"].str.contains("Hybrid Test")].copy()
        if hybrid_df.empty:
            return
        
        def extract_t(name):
            try:
                return float(name.split("=")[1].replace(")", ""))
            except:
                return None
        
        hybrid_df["Threshold"] = hybrid_df["Architecture"].apply(extract_t)
        hybrid_df = hybrid_df.dropna(subset=["Threshold"]).sort_values("Threshold")
        if len(hybrid_df) < 2:
            return
        
        print("   -> Generating Threshold Sensitivity Plot...")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_f1 = "#1f77b4"
        color_cost = "#d62728"
        
        sns.lineplot(data=hybrid_df, x="Threshold", y="F1", marker="o",
                    ax=ax1, color=color_f1, label="F1 Score", linewidth=2.5)
        ax1.set_xlabel("Confidence Threshold ($\\delta$)", fontweight='bold', fontsize=11)
        ax1.set_ylabel("F1 Score (%)", color=color_f1, fontweight='bold', fontsize=11)
        ax1.tick_params(axis='y', labelcolor=color_f1)
        ax1.set_ylim(min(hybrid_df["F1"])*0.98, max(hybrid_df["F1"])*1.02)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xticks(hybrid_df["Threshold"].unique())
        
        ax2 = ax1.twinx()
        sns.lineplot(data=hybrid_df, x="Threshold", y="Cost ($)", marker="s",
                    ax=ax2, color=color_cost, label="Avg. Cost ($)",
                    linewidth=2.5, linestyle="--")
        ax2.set_ylabel("Avg. Cost per Query ($)", color=color_cost,
                      fontweight='bold', fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color_cost)
        ax2.grid(False)
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True)
        if ax2.get_legend():
            ax2.get_legend().remove()
        
        plt.title("Threshold Calibration: Intelligence and Cost Trade-off ", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_threshold_calibration.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Threshold plot error: {e}")


def generate_efficiency_frontier_plot(df_clean: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate efficiency frontier (Pareto curve) plot.
    
    Args:
        df_clean: DataFrame with metrics
        output_dir: Directory to save plot
    """
    print("   -> Generating Efficiency Frontier Plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.scatterplot(
        data=df_clean, x="Cost ($)", y="F1",
        hue="Architecture", style="Architecture", s=200,
        palette="deep", ax=ax
    )
    
    add_smart_labels(ax, df_clean, "Cost ($)", "F1", "Architecture")
    
    plt.title("The Efficiency Frontier: Performance vs. Cost")
    plt.xlabel("Avg. Cost per Query ($)")
    plt.ylabel("F1 Score (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, title="Architecture")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_efficiency_frontier.png")
    plt.close()


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main() -> None:
    """
    Execute complete evaluation pipeline.
    
    This function:
    1. Loads all result JSON files
    2. Computes aggregate metrics
    3. Generates comparative table
    4. Creates 6 visualizations
    5. Exports CSV summary
    """
    print(f"📊 Evaluating Results in {config.LOG_DIR}...\n")
    
    table_data = []
    all_results_map = {}
    
    # Load all results
    for arch_name, filename in ARCHITECTURES:
        file_path = config.LOG_DIR / filename
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
            
            all_results_map[arch_name] = results
            metrics = evaluate_run(results)
            
            entry = {
                "Architecture": arch_name,
                "F1": metrics['f1'] * 100,
                "EM": metrics['em'] * 100,
                "Recall": metrics['recall'] * 100,
                "Latency (s)": metrics['latency'],
                "Cost ($)": metrics['cost'],
                "Steps": metrics['steps'],
            }
            
            if "Hybrid" in arch_name:
                entry["Escalation %"] = metrics.get('escalation_rate', 0.0)
            else:
                entry["Escalation %"] = 0.0
            
            table_data.append(entry)
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    if not table_data:
        print("❌ No data found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Filter main architectures for clean comparison
    valid_names = [
        "Baseline (4o-mini)", "Baseline (4o)",
        "ReAct", "Supervisor", "Self-Correcting", "Plan-Execute", "Network",
        MAIN_HYBRID_LABEL
    ]
    df_clean = df[df["Architecture"].isin(valid_names)].copy()
    
    # Print table
    print(df_clean.sort_values("F1", ascending=False).round(2).to_markdown(index=False))
    
    # Save CSV
    df.to_csv(config.LOG_DIR / "final_metrics_summary.csv", index=False)
    
    # Generate all plots
    generate_effectiveness_plot(df_clean, config.LOG_DIR)
    generate_speed_greed_plot(df_clean, config.LOG_DIR)
    generate_step_stacked_bar(all_results_map, config.LOG_DIR)
    generate_variance_plot(all_results_map, config.LOG_DIR)
    generate_threshold_plot(df, config.LOG_DIR)
    generate_efficiency_frontier_plot(df_clean, config.LOG_DIR)
    
    print(f"\n✅ All plots saved to {config.LOG_DIR}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Run evaluation from command line.
    
    Usage:
        python -m src_thesis.evaluate_results
        
    Outputs:
        - final_metrics_summary.csv
        - 6 PNG visualization files
    """
    main()