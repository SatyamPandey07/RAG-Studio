from typing import List, Dict, Any

class HybridSearch:
    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Merges results from dense vector search and sparse keyword search
        using the Reciprocal Rank Fusion (RRF) algorithm.
        
        Assumes input results are lists of dicts containing a unique 'chunk_id' or 'id'.
        """
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}
        
        # Populate dense ranks
        for rank, item in enumerate(dense_results):
            chunk_id = str(item.get("id") or item.get("chunk_id"))
            chunks_map[chunk_id] = item
            
            # Rank is 0-indexed, add 1 for standard 1-based RRF formula
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Populate sparse ranks
        for rank, item in enumerate(sparse_results):
            chunk_id = str(item.get("id") or item.get("chunk_id"))
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = item
                
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Sort chunks by RRF score in descending order
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_results = []
        for idx, cid in enumerate(sorted_chunk_ids):
            item = chunks_map[cid].copy()
            item["rrf_score"] = rrf_scores[cid]
            item["fused_rank"] = idx + 1
            fused_results.append(item)
            
        return fused_results

    @staticmethod
    def fuse_scores_weighted(
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combines search scores directly using weighted linear interpolation.
        Scores must be normalized between 0.0 and 1.0.
        """
        fused_map: Dict[str, Dict[str, Any]] = {}
        
        for item in dense_results:
            cid = str(item.get("id") or item.get("chunk_id"))
            score = float(item.get("score") or 0.0)
            
            fused_map[cid] = {
                **item,
                "score": score * dense_weight
            }
            
        for item in sparse_results:
            cid = str(item.get("id") or item.get("chunk_id"))
            score = float(item.get("score") or 0.0)
            
            if cid in fused_map:
                fused_map[cid]["score"] += score * sparse_weight
            else:
                fused_map[cid] = {
                    **item,
                    "score": score * sparse_weight
                }
                
        sorted_items = sorted(fused_map.values(), key=lambda x: x["score"], reverse=True)
        return sorted_items
