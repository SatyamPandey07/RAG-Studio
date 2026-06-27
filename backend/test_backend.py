import sys
import os

# Add parent path to import app services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.search import HybridSearch

def test_chunker():
    print("Testing Chunker...")
    text = "Hello world! This is a test document to ensure chunking works correctly. RAG Studio is great."
    
    # Test Fixed Chunker
    fixed_chunks = Chunker.chunk_document(text, "fixed", 20, 5)
    print(f"  Fixed chunks generated: {len(fixed_chunks)}")
    assert len(fixed_chunks) > 0, "Fixed chunker failed to produce chunks"
    
    # Test Recursive Chunker
    rec_chunks = Chunker.chunk_document(text, "recursive", 30, 5)
    print(f"  Recursive chunks generated: {len(rec_chunks)}")
    assert len(rec_chunks) > 0, "Recursive chunker failed to produce chunks"
    
    # Test Semantic Chunker
    sem_chunks = Chunker.chunk_document(text, "semantic", 50, 0)
    print(f"  Semantic chunks generated: {len(sem_chunks)}")
    assert len(sem_chunks) > 0, "Semantic chunker failed to produce chunks"
    print("✓ Chunker tests passed successfully.")

def test_embedder():
    print("\nTesting Embedder...")
    text = "Test document embedding"
    
    # Test Google Embedder Mock Fallback
    vector = Embedder.get_embedding(text, "google", "text-embedding-004")
    print(f"  Mock vector dimensions: {len(vector)}")
    assert len(vector) == 768, f"Expected 768 dimensions, got {len(vector)}"
    
    # Test OpenAI Embedder Mock Fallback
    vector_oa = Embedder.get_embedding(text, "openai", "text-embedding-3-small")
    print(f"  Mock OpenAI vector dimensions: {len(vector_oa)}")
    assert len(vector_oa) == 1536, f"Expected 1536 dimensions, got {len(vector_oa)}"
    print("✓ Embedder tests passed successfully.")

def test_hybrid_search():
    print("\nTesting Hybrid Search RRF Fusion...")
    
    dense = [
        {"id": "doc1", "score": 0.9, "text": "RAG is awesome"},
        {"id": "doc2", "score": 0.8, "text": "Vector DB is helpful"}
    ]
    
    sparse = [
        {"id": "doc3", "score": 0.95, "text": "Keyword match"},
        {"id": "doc1", "score": 0.7, "text": "RAG is awesome"}
    ]
    
    fused = HybridSearch.reciprocal_rank_fusion(dense, sparse, k=60)
    print(f"  Fused results count: {len(fused)}")
    
    # doc1 is present in both, should rank high
    assert fused[0]["id"] == "doc1", f"Expected doc1 to be top rank, got {fused[0]['id']}"
    print("✓ Hybrid search RRF tests passed successfully.")

if __name__ == "__main__":
    print("--- Running RAG Studio Backend Verification Tests ---")
    try:
        test_chunker()
        test_embedder()
        test_hybrid_search()
        print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
    except AssertionError as e:
        print(f"\n❌ Test verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error running tests: {e}")
        sys.exit(1)
