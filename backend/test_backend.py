import sys
import os

# Add parent path to import app services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.chunker import Chunker
from app.services.embedder import EmbeddingService
from app.services.search import HybridSearch
from app.services.reranker import Reranker
from app.services.evaluator import Evaluator
from app.services.document_parser import DocumentParser
from app.services.llm import LLMService

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
    print("\nTesting EmbeddingService...")
    text = "Test document embedding"
    
    # Test Google Embedder Mock Fallback
    vector = EmbeddingService.embed_query(text, "google")
    print(f"  Mock vector dimensions: {len(vector)}")
    assert len(vector) == 768, f"Expected 768 dimensions, got {len(vector)}"
    
    # Test OpenAI Embedder Mock Fallback
    vector_oa = EmbeddingService.embed_query(text, "openai")
    print(f"  Mock OpenAI vector dimensions: {len(vector_oa)}")
    assert len(vector_oa) == 1536, f"Expected 1536 dimensions, got {len(vector_oa)}"
    print("✓ EmbeddingService tests passed successfully.")

def test_hybrid_search():
    print("\nTesting Hybrid Search & BM25 index...")
    
    chunks = [
        {"id": "c1", "text": "RAG retrieves relevant snippets from documents.", "source": "doc1.txt"},
        {"id": "c2", "text": "Vector databases store dense vector representations.", "source": "doc2.txt"},
        {"id": "c3", "text": "Keyword search parses exact word matches using BM25.", "source": "doc3.txt"},
    ]
    
    # Build BM25 index
    collection_name = "test_collection"
    HybridSearch.build_bm25_index(collection_name, chunks)
    assert HybridSearch.has_index(collection_name), "BM25 index not registered"
    
    # Sparse search
    sparse_res = HybridSearch.sparse_search(collection_name, "vector databases", top_k=2)
    print(f"  Sparse results count: {len(sparse_res)}")
    assert len(sparse_res) > 0, "Sparse search returned no results"
    assert "vector" in sparse_res[0]["text"].lower(), "Top result should be related to vector databases"
    
    # RRF Fusion
    dense_res = [
        {"id": "c2", "score": 0.9, "text": "Vector databases store dense vector representations.", "source": "doc2.txt"},
        {"id": "c1", "score": 0.6, "text": "RAG retrieves relevant snippets from documents.", "source": "doc1.txt"},
    ]
    
    fused = HybridSearch.reciprocal_rank_fusion(dense_res, sparse_res, dense_weight=0.7, sparse_weight=0.3)
    print(f"  Fused results count: {len(fused)}")
    assert len(fused) > 0, "Hybrid fusion returned no results"
    print("✓ Hybrid search tests passed successfully.")

def test_reranker():
    print("\nTesting Reranker...")
    query = "What is RAG?"
    results = [
        {"id": "c2", "text": "Vector databases store dense vector representations.", "score": 0.8},
        {"id": "c1", "text": "RAG retrieves relevant snippets from documents.", "score": 0.6},
    ]
    reranked = Reranker.rerank(query, results, top_n=2)
    print(f"  Top reranked item: {reranked[0]['text']}")
    # RAG should rank first because of higher term overlap
    assert "RAG" in reranked[0]["text"], "Reranker should boost RAG-related chunk to top"
    print("✓ Reranker tests passed successfully.")

def test_document_parser():
    print("\nTesting DocumentParser...")
    sample_text = "This is a simple plain text document with some words."
    file_bytes = sample_text.encode("utf-8")
    text, meta = DocumentParser.parse("sample.txt", file_bytes)
    print(f"  Parsed text length: {len(text)}, word count: {meta.get('word_count')}")
    assert text == sample_text, "Parsed text mismatch"
    assert meta.get("word_count") == 10, f"Expected 10 words, got {meta.get('word_count')}"
    print("✓ DocumentParser tests passed successfully.")

def test_llm_service():
    print("\nTesting LLMService...")
    query = "What is chunking?"
    context = [{"text": "Chunking splits text into smaller segments.", "source": "doc.txt", "score": 0.9}]
    result = LLMService.chat(query, context, "System prompt")
    print(f"  Simulated Answer: {result['answer']}")
    assert "answer" in result, "LLM chat response missing answer"
    assert result["demo_mode"] is True, "Expected demo_mode to be True"
    print("✓ LLMService tests passed successfully.")

def test_evaluator():
    print("\nTesting Evaluator...")
    query = "What is RAG?"
    answer = "RAG is retrieval-augmented generation."
    context = [{"text": "RAG is retrieval-augmented generation that retrieves chunks.", "source": "doc.txt", "score": 0.95}]
    
    # Generation eval
    gen_metrics = Evaluator.evaluate_generation(query, answer, context)
    print(f"  Faithfulness: {gen_metrics['faithfulness']}")
    print(f"  Groundedness: {gen_metrics['groundedness']}")
    assert gen_metrics["faithfulness"] > 0.5, "Faithfulness should be high"
    
    # Retrieval eval
    ret_chunks = [{"id": "c1", "score": 0.9}]
    ret_metrics = Evaluator.evaluate_retrieval(ret_chunks, relevant_doc_ids=["c1"])
    print(f"  Recall@K: {ret_metrics['recall_at_k']}")
    assert ret_metrics["recall_at_k"] > 0, "Recall should be greater than 0"
    print("✓ Evaluator tests passed successfully.")

if __name__ == "__main__":
    print("--- Running RAG Studio Backend Verification Tests ---")
    try:
        test_chunker()
        test_embedder()
        test_hybrid_search()
        test_reranker()
        test_document_parser()
        test_llm_service()
        test_evaluator()
        print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
    except AssertionError as e:
        print(f"\n❌ Test verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error running tests: {e}")
        sys.exit(1)
