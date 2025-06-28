from sentence_transformers import SentenceTransformer , util
import torch
from src.company_data import clean_resume_text

def recommend_candidates_from_job_description1(cvs,
                                             job_description,
                                             top_k=5):
    
    model = SentenceTransformer('E:/vs codes/REC/job_embeddings_model_evaluated2')
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    # Embed all provided CV texts
    filenames = []
    cv_embeddings = []
    for cv in cvs:
        filenames.append(cv.user_id)
        cv_text = cv.text
        cleaned_text = clean_resume_text(cv_text)  # Clean the CV text
        cv_embeddings.append(model.encode(cleaned_text, convert_to_tensor=True))

    if not cv_embeddings:
        return []

    # Stack embeddings into a tensor
    cv_embeddings_tensor = torch.stack(cv_embeddings)

    # Calculate similarity
    cosine_scores = util.cos_sim(job_embedding, cv_embeddings_tensor)[0]
    actual_top_k = min(top_k, len(filenames))
    top_results = torch.topk(cosine_scores, k=actual_top_k)

    # Format results
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        candidate_key = filenames[idx.item()]
        results.append({
            "user_id": candidate_key,
            "similarity": round(score.item(), 4)
        })

    return results