from sentence_transformers import SentenceTransformer , util
import torch
from utils.hybridembedder import HybridEmbedder

from utils.hybridembedder import HybridEmbedder

def recommend_candidates_from_job_description1(cvs,
                              job_description,
                              top_k=5,
                              k_constant=5):
    if not cvs:
        return []
        
    # Initialize hybrid embedder with RRF constant
    embedder = HybridEmbedder(k_constant=k_constant)
    
    # Extract CV texts and user IDs
    cv_texts = [cv.text for cv in cvs]
    user_ids = [cv.user_id for cv in cvs]
    
    # Use batch processing for better performance
    all_scores = embedder.batch_rank_cvs(job_description, cv_texts)
    
    # Create candidate objects with scores
    candidates = [
        {
            'user_id': user_id,
            'similarity': float(scores['hybrid']),
            'scores': scores  
        }
        for user_id, scores in zip(user_ids, all_scores)
    ]
    
    # Sort by hybrid similarity and take top-k
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Format results for the top-k candidates
    results = [
        {
            "user_id": candidate['user_id'],
            "similarity": round(candidate['similarity'], 4)
        }
        for candidate in candidates[:min(top_k, len(candidates))]
    ]
    
    return results