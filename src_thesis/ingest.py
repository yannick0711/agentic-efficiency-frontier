"""
Wikipedia data ingestion module for vector database indexing.

This module handles the embedding and indexing of Wikipedia abstracts
into a Qdrant vector database for semantic search.

Key Features:
    - Batch processing with checkpointing for large datasets
    - Automatic retry on rate limits
    - Memory-efficient streaming ingestion
    - Progress tracking and RAM monitoring

The ingestion process:
    1. Read Wikipedia abstracts (5.2M documents)
    2. Generate embeddings via OpenAI API
    3. Upload to Qdrant with UUID-based deduplication
    4. Save checkpoints for resumability

Example:
    >>> from src_thesis.ingest import ingest_wiki
    >>> ingest_wiki()
    Indexing 5,233,329 documents...
    [Progress bar]
    ✅ Indexing complete!
"""

import uuid
import time
import ujson
import gc
import resource
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from . import config
from .llm_client import create_embeddings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Batch size for embedding and upload
BATCH_SIZE: int = 1000

# Sleep duration between batches to avoid rate limits
RATE_LIMIT_SLEEP: int = 5

# Checkpoint file for resuming interrupted ingestion
CHECKPOINT_FILE: Path = Path("ingest_checkpoint.txt")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_ram_gb() -> float:
    """
    Get current RAM usage of the process in gigabytes.
    
    Returns:
        RAM usage in GB
        
    Note:
        On macOS (BSD), ru_maxrss is in kilobytes.
        On Linux, it may be in different units.
    """
    kb_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_usage / 1024 / 1024


def get_start_line() -> int:
    """
    Read checkpoint file to determine resume point.
    
    Returns:
        Line number to resume from (0 if no checkpoint)
    """
    if CHECKPOINT_FILE.exists():
        try:
            return int(CHECKPOINT_FILE.read_text().strip())
        except (ValueError, FileNotFoundError):
            return 0
    return 0


def save_checkpoint(line_num: int) -> None:
    """
    Save current progress to checkpoint file.
    
    Args:
        line_num: Current line number in the dataset
    """
    CHECKPOINT_FILE.write_text(str(line_num))


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db() -> QdrantClient:
    """
    Initialize connection to Qdrant and create collection if needed.
    
    Returns:
        Connected Qdrant client instance
        
    Note:
        Creates collection with:
        - Cosine similarity distance metric
        - On-disk storage for large datasets
        - INT8 quantization for memory efficiency
    """
    print(f"🔌 Initializing connection to Qdrant ({config.QDRANT_URL})...")
    
    # Connect to Qdrant server
    client = QdrantClient(url=config.QDRANT_URL)
    
    # Create collection if it doesn't exist
    if not client.collection_exists(config.COLLECTION_NAME):
        print(f"📦 Creating collection '{config.COLLECTION_NAME}'...")
        
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
                on_disk=True  # Store on disk for huge datasets
            ),
            # Quantization reduces memory usage
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=False
                )
            )
        )
        print(f"✅ Collection created successfully")
    else:
        print(f"✅ Collection '{config.COLLECTION_NAME}' already exists")
    
    return client


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def flush_batch(
    client: QdrantClient,
    texts: List[str],
    payloads: List[Dict[str, Any]]
) -> None:
    """
    Embed and upload a batch of documents.
    
    This function:
    1. Generates embeddings for all texts in batch
    2. Creates Qdrant points with UUIDs
    3. Uploads to the database
    
    Args:
        client: Connected Qdrant client
        texts: List of document texts to embed
        payloads: List of metadata dicts (id, title, text)
        
    Raises:
        Exception: If embedding or upload fails
    """
    try:
        # Step 1: Generate embeddings
        response = create_embeddings(texts, model=config.EMBED_MODEL)
        embeddings = [item.embedding for item in response.data]
        
        # Step 2: Create points with UUIDs
        points = []
        for embedding, metadata in zip(embeddings, payloads):
            # Generate deterministic UUID from document ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, metadata['id']))
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
            )
        
        # Step 3: Upload to Qdrant (async upload, don't wait)
        client.upload_points(
            collection_name=config.COLLECTION_NAME,
            points=points,
            wait=False
        )
        
    except Exception as e:
        print(f"❌ Batch upload failed: {e}")
        raise


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

def ingest_wiki() -> None:
    """
    Main ingestion pipeline for Wikipedia abstracts.
    
    Pipeline:
        1. Initialize Qdrant connection
        2. Load checkpoint (if resuming)
        3. Stream Wikipedia file line by line
        4. Batch documents and flush to database
        5. Save checkpoints periodically
        6. Monitor RAM usage
        
    The function processes 5.2M documents in batches of 1000,
    with automatic checkpointing for crash recovery.
    
    Example:
        >>> ingest_wiki()
        🚀 Starting fresh...
        Reading /path/to/wiki_abstracts.jsonl...
        Indexing | RAM: 2.34 GB: 100%|████████| 5233329/5233329
        ✅ Indexing complete!
    """
    # Step 1: Initialize database
    client = init_db()
    
    # Step 2: Check for Wikipedia file
    if not config.WIKI_DUMP_FILE.exists():
        print(f"❌ ERROR: Wiki file not found at {config.WIKI_DUMP_FILE}")
        print("Please download the Wikipedia dump first.")
        return
    
    # Step 3: Load checkpoint
    start_line = get_start_line()
    if start_line > 0:
        print(f"🔄 Resuming from line {start_line:,}...")
    else:
        print("🚀 Starting fresh ingestion...")
    
    # Step 4: Initialize batch containers
    batch_docs = []
    batch_texts = []
    
    # Estimated total documents (approximate)
    TOTAL_ESTIMATE = 5_200_000
    
    print(f"📖 Reading {config.WIKI_DUMP_FILE}...")
    
    # Initialize progress bar
    pbar = tqdm(
        total=TOTAL_ESTIMATE,
        initial=start_line,
        desc="Indexing"
    )
    
    # Step 5: Stream and process documents
    with open(config.WIKI_DUMP_FILE, "r", encoding="utf-8") as f:
        # Fast-forward to checkpoint
        if start_line > 0:
            for _ in range(start_line):
                next(f, None)
        
        # Main processing loop
        for i, line in enumerate(f, start=start_line):
            try:
                # Parse document
                doc = ujson.loads(line)
                doc_id = doc['id']
                raw_text = doc.get('text', "")
                
                # Normalize text field (can be list or string)
                text = (
                    " ".join(raw_text)
                    if isinstance(raw_text, list)
                    else str(raw_text)
                )
                
                # Skip empty documents
                if not text.strip():
                    pbar.update(1)
                    continue
                
                # Add to batch
                batch_docs.append({
                    "id": doc_id,
                    "text": text,
                    "title": doc['title']
                })
                batch_texts.append(text)
                
                # Flush batch when full
                if len(batch_docs) >= BATCH_SIZE:
                    flush_batch(client, batch_texts, batch_docs)
                    save_checkpoint(i + 1)
                    
                    # Clear batch
                    batch_docs = []
                    batch_texts = []
                    
                    # Memory management
                    gc.collect()  # Force garbage collection
                    
                    # Update progress bar with RAM usage
                    ram_usage = get_current_ram_gb()
                    pbar.set_description(f"Indexing | RAM: {ram_usage:.2f} GB")
                    
                    # Rate limiting
                    time.sleep(RATE_LIMIT_SLEEP)
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n⚠️ Skipping line {i}: {e}")
                continue
        
        # Flush any remaining documents
        if batch_docs:
            flush_batch(client, batch_texts, batch_docs)
            save_checkpoint(i + 1)
    
    pbar.close()
    print("\n✅ Indexing complete!")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    """
    Run Wikipedia ingestion from command line.
    
    Usage:
        python -m src_thesis.ingest
        
    The script will:
    - Resume from checkpoint if interrupted
    - Display progress with RAM monitoring
    - Save checkpoints every 1000 documents
    
    Estimated time: 3-4 hours for 5.2M documents
    Estimated cost: ~$100-150 for embeddings
    """
    ingest_wiki()