from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import pickle
from utils.AI_agent import extract_and_check
model = SentenceTransformer('E:/vs codes/REC/job_embeddings_model_evaluated2')

Data = pd.read_csv("E:/vs codes/REC/data/job_lists.csv", delimiter=";")

with open('E:/vs codes/REC/job_embeddings3.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

def recommend_jobs(skills , top_k = None):
    skills_extracted = extract_and_check(skills)
    skills_embedding = model.encode(skills_extracted, convert_to_tensor=True)
    cosine_scores = util.cos_sim(skills_embedding, loaded_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    
    scores = []
    ids = []
    
    for score , idx in zip(top_results.values, top_results.indices):
        
        idx_value = idx.item()
        scores.append(round(score.item(), 4))
        ids1 = Data["id"].loc[idx_value]
        ids.append(ids1)

    df = pd.DataFrame({
        "id": ids,
        "score": scores
    })

    return df.to_dict(orient="records")