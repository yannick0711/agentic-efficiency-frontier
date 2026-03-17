"""
Configuration module for the Agentic RAG thesis project.

This module centralizes all configuration parameters including:
- File paths for data and logs
- Database connection settings
- Model specifications
- API credentials

The module uses python-dotenv for environment variable management,
allowing secure storage of API keys outside the codebase.

Example:
    >>> from src_thesis import config
    >>> print(config.LLM_MODEL)
    'gpt-4o-mini'
    >>> print(config.LOG_DIR)
    Path('/home/user/thesis/logs/latest')
"""

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (parent of src_thesis/)
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Data directory for datasets and embeddings
DATA_DIR: Final[Path] = BASE_DIR / "data"

# Dynamic logging directory
# If THESIS_RUN_ID is set (by run_all.py), use that folder
# Otherwise, default to "latest" for manual runs
RUN_ID: Final[str] = os.getenv("THESIS_RUN_ID", "latest")
LOG_DIR: Final[Path] = BASE_DIR / "logs" / RUN_ID

# Ensure required directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA FILE PATHS
# =============================================================================

# Wikipedia abstracts dump (5.2M documents)
WIKI_DUMP_FILE: Final[Path] = DATA_DIR / "wiki_abstracts.jsonl"

# HotpotQA dataset files
HOTPOT_TRAIN_FILE: Final[Path] = DATA_DIR / "hotpot_train_v1.1.json"
TEST_DATA_FILE: Final[Path] = DATA_DIR / "hotpot_eval_1000.json"


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Qdrant vector database connection (Docker container)
QDRANT_URL: Final[str] = "http://localhost:6333"

# Collection name for Wikipedia embeddings
COLLECTION_NAME: Final[str] = "hotpot_wiki"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# OpenAI embedding model for semantic search
# text-embedding-3-small: 1536 dimensions, cost-effective
EMBED_MODEL: Final[str] = "text-embedding-3-small"

# OpenAI chat model for agents
# gpt-4o-mini: Small, fast, cost-efficient for experiments
# gpt-4o: Large, expensive, used as baseline comparison
LLM_MODEL: Final[str] = "gpt-4o-mini"

# Embedding vector dimensionality
EMBED_DIM: Final[int] = 1536


# =============================================================================
# API CREDENTIALS
# =============================================================================

# OpenAI API key (loaded from .env file)
# Required for embeddings and chat completions
OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

# Maximum parallel workers for experiments
# Can be overridden via MAX_WORKERS environment variable
MAX_WORKERS: Final[int] = int(os.getenv("MAX_WORKERS", "10"))


# =============================================================================
# DEBUGGING / DEVELOPMENT
# =============================================================================

# Enable verbose logging (set THESIS_DEBUG=1 in .env)
DEBUG_MODE: Final[bool] = os.getenv("THESIS_DEBUG", "0") == "1"


# Print configuration summary when module is imported
if __name__ == "__main__" or DEBUG_MODE:
    print("=" * 60)
    print("THESIS CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Base Directory:     {BASE_DIR}")
    print(f"Data Directory:     {DATA_DIR}")
    print(f"Log Directory:      {LOG_DIR}")
    print(f"Run ID:             {RUN_ID}")
    print(f"Qdrant URL:         {QDRANT_URL}")
    print(f"Collection:         {COLLECTION_NAME}")
    print(f"Embedding Model:    {EMBED_MODEL}")
    print(f"LLM Model:          {LLM_MODEL}")
    print(f"Max Workers:        {MAX_WORKERS}")
    print(f"API Key Loaded:     {'✓' if OPENAI_API_KEY else '✗'}")
    print("=" * 60)