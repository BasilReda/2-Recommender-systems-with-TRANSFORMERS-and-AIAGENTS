from sentence_transformers import SentenceTransformer , util
import torch
from src.company_data import (
    extract_text_description,
    clean_resume_dataframe,
)
from src.rec_helper import  load_embedded_cv

df = extract_text_description(path = "E:/vs codes/REC/cv")

cleaned_resumes = clean_resume_dataframe(df)

embedded_cv = load_embedded_cv( file_path='E:/vs codes/REC/embedded_evaluated2_cv.pkl' )

def recommend_candidates_from_job_description(job_description = "senior data analyst",
                                              relevant_cvs = None,
                                              top_k=5):
    
    model = SentenceTransformer('E:/vs codes/REC/job_embeddings_model_evaluated2')
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    ## loop on the cv i have to check if depend on its name if its relevant for the job or no
    filtered_cv_embeddings = {}
    for cv_name in relevant_cvs:
        if cv_name in embedded_cv:
            filtered_cv_embeddings[cv_name] = embedded_cv[cv_name]

    cv_keys = list(filtered_cv_embeddings.keys())
    cv_embedding_list = list(filtered_cv_embeddings.values())

    if not cv_embedding_list:
        return []  

    cv_embeddings_tensor = torch.tensor([emb for emb in cv_embedding_list])
    
    # Calculate similarity
    cosine_scores = util.cos_sim(job_embedding, cv_embeddings_tensor)[0]

    # Adjust top_k if we have fewer CVs than requested
    actual_top_k = min(top_k, len(cv_keys))

    top_results = torch.topk(cosine_scores, k=actual_top_k)
    
    # Format results
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        candidate_key = cv_keys[idx.item()]
        results.append({
            "filename": candidate_key,
            "similarity": round(score.item(), 4)
        })
    
    return results
