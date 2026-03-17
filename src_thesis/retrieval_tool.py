"""
Vector search interface for semantic document retrieval.

This module provides the interface to Qdrant vector database for
semantic search over Wikipedia abstracts. It handles:
- Query embedding generation
- Vector similarity search
- Result formatting for LLM consumption

The RetrievalTool class encapsulates all search logic with automatic
retry and timeout handling.

Example:
    >>> from src_thesis.retrieval_tool import search_wiki
    >>> results = search_wiki("capital of France", k=3)
    >>> print(results)
    '--- Document 1 ---\nTitle: Paris\n...'
"""

from typing import List, Set
from qdrant_client import QdrantClient, models

from . import config
from .llm_client import create_embeddings


# =============================================================================
# RETRIEVAL TOOL CLASS
# =============================================================================

class RetrievalTool:
    """
    Interface to Qdrant vector database for semantic search.
    
    This class manages connections to the vector database and provides
    methods for searching and formatting results.
    
    Attributes:
        client: Qdrant client instance
        collection_name: Name of the vector collection to search
        embed_model: OpenAI embedding model name
        timeout: Connection timeout in seconds
        
    Example:
        >>> tool = RetrievalTool()
        >>> results = tool.search("Paris", k=3)
        >>> print(results)
    """
    
    def __init__(
        self,
        url: str = config.QDRANT_URL,
        collection_name: str = config.COLLECTION_NAME,
        embed_model: str = config.EMBED_MODEL,
        timeout: float = 60.0
    ):
        """
        Initialize connection to vector database.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of the collection to search
            embed_model: OpenAI embedding model
            timeout: Connection timeout in seconds (increased from default 5s)
        """
        self.collection_name = collection_name
        self.embed_model = embed_model
        
        # Initialize client with extended timeout to prevent crashes
        self.client = QdrantClient(url=url, timeout=timeout)
    
    def search(self, query: str, k: int = 5) -> str:
        """
        Perform semantic search and return formatted results.
        
        This method:
        1. Embeds the query using OpenAI embeddings
        2. Searches the vector database for similar documents
        3. Formats results for LLM consumption
        
        Args:
            query: The search query text
            k: Number of documents to retrieve
            
        Returns:
            Formatted string with document titles, relevance scores, and content
            
        Raises:
            AttributeError: If search API is unavailable (falls back to query_points)
            
        Example:
            >>> tool = RetrievalTool()
            >>> results = tool.search("Who invented the telephone?", k=3)
            >>> print(results[:100])
            '--- Document 1 ---\nTitle: Alexander Graham Bell\n...'
        """
        # Step 1: Generate query embedding
        embedding_response = create_embeddings([query], model=self.embed_model)
        query_vector = embedding_response.data[0].embedding
        
        # Step 2: Search vector database
        # Try standard search API first, fallback to query_points if unavailable
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k
            )
        except AttributeError:
            # Fallback for older Qdrant versions
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k
            )
            results = response.points
        
        # Step 3: Format results for LLM
        return self._format_results(results)
    
    def _format_results(self, results: List) -> str:
        """
        Format search results for LLM consumption.
        
        Creates a structured text format that helps the LLM:
        1. Identify document sources for citation
        2. Assess relevance via scores
        3. Extract answer entities from content
        
        Args:
            results: List of search result objects from Qdrant
            
        Returns:
            Formatted string with numbered documents
            
        Example output:
            --- Document 1 ---
            Title: Paris
            Relevance Score: 0.8934
            Content: Paris is the capital and most populous city...
        """
        if not results:
            return "No relevant documents found."
        
        formatted_output = []
        
        for i, result in enumerate(results):
            # Extract metadata from result
            title = result.payload.get('title', 'Unknown Title')
            text = result.payload.get('text', '').strip()
            score = result.score
            
            # Format as structured block
            entry = (
                f"--- Document {i+1} ---\n"
                f"Title: {title}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content: {text}\n"
            )
            formatted_output.append(entry)
        
        return "\n".join(formatted_output)


# =============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# =============================================================================

# Global instance for backward compatibility
_default_tool = RetrievalTool()


def search_wiki(query: str, k: int = 5) -> str:
    """
    Search Wikipedia abstracts using default retrieval tool.
    
    This is a convenience function for backward compatibility.
    For new code, prefer using RetrievalTool directly.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        Formatted search results string
        
    Example:
        >>> results = search_wiki("capital of France", k=3)
        >>> "Paris" in results
        True
    """
    return _default_tool.search(query, k=k)


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the retrieval tool with a simple query.
    
    This requires:
    - Qdrant running on localhost:6333
    - Collection 'hotpot_wiki' with indexed documents
    """
    print("Testing Retrieval Tool...")
    
    try:
        # Test with a simple query
        tool = RetrievalTool()
        results = search_wiki("history", k=2)
        
        print("Search Results:")
        print(results)
        print("\n✅ Retrieval tool works!")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        print("\nMake sure:")
        print("1. Qdrant is running: docker-compose up -d")
        print("2. Collection exists: check src_thesis/ingest.py")