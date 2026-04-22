import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.company_data import clean_resume_text


def load_embedded_cv(file_path='embedded_evaluated2_cv.pkl'):
    with open(file_path , "rb") as f:
        embedded_cv = pickle.load(f)

    return embedded_cv

# def recommend_candidates(dataframe  , job_description  , top_k= None , embeddings_dict = None ):
    
#     # Load pre-trained sentence embedding model
#     # model = SentenceTransformer('all-MiniLM-L6-v2')  # General-purpose model
#     # model = SentenceTransformer('job_embeddings_model_evaluated')  
#     model = SentenceTransformer('job_embeddings_model_evaluated2')  # Fast lightweight model
    
#     # Clean the job description using the same function used for resumes
#     cleaned_job_desc = clean_resume_text(job_description)
    
#     # Get embeddings for job description and all resumes
#     job_embedding = model.encode([cleaned_job_desc])[0]
    
#     # Store similarity scores
#     similarities = []
    
#     # Calculate similarity for each resume
#     for idx, row in dataframe.iterrows():
#         filename = row['filename']
#         resume_text = row['Resume']
        
#         if embeddings_dict and filename in embeddings_dict:
#             resume_embedding = embeddings_dict[filename]
#         # Calculate cosine similarity
#         similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
#         similarities.append({
#             'Name': row['filename'],
#             'Similarity': similarity,
#             'Resume': resume_text
#         })
    
#     # Convert to DataFrame and sort by similarity
#     results_df = pd.DataFrame(similarities)
#     results_df = results_df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    
#     # Add rank column
#     results_df['Rank'] = results_df.index + 1
    
#     # Return top_k results
#     return results_df.head(top_k)[['Rank', 'Name', 'Similarity', 'Resume']]