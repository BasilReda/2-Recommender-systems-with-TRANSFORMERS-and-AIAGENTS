from sentence_transformers import SentenceTransformer , util
import torch
from utils.hybridembedder import HybridEmbedder

# def recommend_candidates_from_job_description1(cvs,
#                                              job_description,
#                                              top_k=5):
    
#     model = SentenceTransformer('E:/vs codes/REC/job_embeddings_model_evaluated2')
#     job_embedding = model.encode(job_description, convert_to_tensor=True)

#     # Embed all provided CV texts
#     filenames = []
#     cv_embeddings = []
#     for cv in cvs:
#         filenames.append(cv.user_id)
#         cv_text = cv.text
#         cleaned_text = clean_resume_text(cv_text)  # Clean the CV text
#         accurate_text = extract_and_check(cleaned_text)
#         cv_embeddings.append(model.encode(accurate_text, convert_to_tensor=True))

#     if not cv_embeddings:
#         return []

#     # Stack embeddings into a tensor
#     cv_embeddings_tensor = torch.stack(cv_embeddings)

#     # Calculate similarity
#     cosine_scores = util.cos_sim(job_embedding, cv_embeddings_tensor)[0]
#     actual_top_k = min(top_k, len(filenames))
#     top_results = torch.topk(cosine_scores, k=actual_top_k)

#     # Format results
#     results = []
#     for score, idx in zip(top_results.values, top_results.indices):
#         candidate_key = filenames[idx.item()]
#         results.append({
#             "user_id": candidate_key,
#             "similarity": round(score.item(), 4)
#         })

#     return results

from utils.hybridembedder import HybridEmbedder

def recommend_candidates_from_job_description1(cvs,
                              job_description,
                              top_k=5,
                              k_constant=5):
    """
    Recommend candidates for a job using hybrid embeddings with Reciprocal Rank Fusion.
    
    Args:
        cvs: List of CV objects with user_id and text attributes
        job_description: Text of the job description
        top_k: Number of top candidates to return
        k_constant: Constant for RRF formula (higher values reduce impact of high ranks)
        
    Returns:
        List of dictionaries with user_id and similarity scores
    """
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
            'scores': scores  # Store all scores for later use
        }
        for user_id, scores in zip(user_ids, all_scores)
    ]
    
    # Sort by hybrid similarity and take top-k
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Format results for the top-k candidates
    results = [
        {
            "user_id": candidate['user_id'],
            "similarity": round(candidate['similarity'], 4),
            "dense_score": round(float(candidate['scores']['dense']), 4),
            "sparse_score": round(float(candidate['scores']['sparse']), 4),
            "normalized_sparse": round(float(candidate['scores']['normalized_sparse']), 4)
        }
        for candidate in candidates[:min(top_k, len(candidates))]
    ]
    
    return results