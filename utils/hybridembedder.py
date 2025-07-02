from sentence_transformers import SentenceTransformer, util
import numpy as np
from fastembed.sparse.bm25 import Bm25
from src.company_data import clean_resume_text
from utils.AI_agent import extract_and_check

class HybridEmbedder:
    """
    Combines dense embeddings from a Sentence Transformer model with 
    sparse embeddings from BM25 for improved text matching using Reciprocal Rank Fusion.
    """
    
    def __init__(self, dense_model_path='E:/vs codes/REC/job_embeddings_model_evaluated2',
                 sparse_model_name="Qdrant/bm25",
                 k_constant=60,
                 use_cache=True):
        """
        Initialize the hybrid embedder with RRF ranking parameters.

        Args:
            dense_model_path: Path to the SentenceTransformer model
            sparse_model_name: Name of the sparse embedding model
            k_constant: Constant used in RRF formula (default: 60, higher values reduce impact of high ranks)
            use_cache: Whether to cache embeddings for reuse (improves performance)
        """
        self.dense_model = SentenceTransformer(dense_model_path)
        self.sparse_model = Bm25(sparse_model_name)
        self.k_constant = k_constant
        self.use_cache = use_cache
        self._cache = {} if use_cache else None

    def encode_job(self, job_description):
        """
        Generate both dense and sparse embeddings for a job description.
        
        Args:
            job_description: Text of the job description
            
        Returns:
            Dictionary with dense and sparse embeddings
        """
        # Check cache
        if self.use_cache:
            cache_key = f"job_{hash(job_description)}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        dense_embedding = self.dense_model.encode(job_description, convert_to_tensor=True)
        sparse_embedding = next(self.sparse_model.query_embed([job_description]))
        
        result = {
            'dense': dense_embedding,
            'sparse': sparse_embedding
        }
        
        # Store in cache
        if self.use_cache:
            self._cache[cache_key] = result
            
        return result
    
    def encode_cv(self, cv_text):
        """
        Generate both dense and sparse embeddings for a CV.
        
        Args:
            cv_text: Text of the CV
            
        Returns:
            Dictionary with dense and sparse embeddings
        """
        # Check cache
        if self.use_cache:
            cache_key = f"cv_{hash(cv_text)}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        cleaned_text = clean_resume_text(cv_text)
        accurate_text = extract_and_check(cleaned_text)
        
        dense_embedding = self.dense_model.encode(accurate_text, convert_to_tensor=True)
        sparse_embedding = next(self.sparse_model.passage_embed([accurate_text]))
        
        result = {
            'dense': dense_embedding,
            'sparse': sparse_embedding
        }
        
        # Store in cache
        if self.use_cache:
            self._cache[cache_key] = result
            
        return result
    
    def batch_encode_cvs(self, cv_texts):
        """
        Efficiently encode multiple CV texts in batch.
        
        Args:
            cv_texts: List of CV texts
            
        Returns:
            List of dictionaries with dense and sparse embeddings
        """
        results = []
        cleaned_texts = []
        
        # First clean and process all texts
        for cv_text in cv_texts:
            if self.use_cache:
                cache_key = f"cv_{hash(cv_text)}"
                if cache_key in self._cache:
                    results.append(self._cache[cache_key])
                    cleaned_texts.append(None)  # Placeholder
                    continue
                    
            cleaned_text = clean_resume_text(cv_text)
            accurate_text = extract_and_check(cleaned_text)
            cleaned_texts.append(accurate_text)
        
        # Only encode texts that weren't cached
        texts_to_encode = [text for text in cleaned_texts if text is not None]
        if texts_to_encode:
            # Batch encode with the dense model for efficiency
            dense_embeddings = self.dense_model.encode(texts_to_encode, convert_to_tensor=True)
            
            # Process each non-cached CV
            j = 0
            for i, cleaned_text in enumerate(cleaned_texts):
                if cleaned_text is not None:
                    dense_embedding = dense_embeddings[j] if isinstance(dense_embeddings, list) else dense_embeddings[j]
                    j += 1
                    
                    # Sparse embeddings still need individual processing
                    sparse_embedding = next(self.sparse_model.passage_embed([cleaned_text]))
                    
                    result = {
                        'dense': dense_embedding,
                        'sparse': sparse_embedding
                    }
                    
                    if self.use_cache:
                        cache_key = f"cv_{hash(cv_texts[i])}"
                        self._cache[cache_key] = result
                        
                    results.append(result)
        
        return results
    
    def calculate_similarity(self, job_embeddings, cv_embeddings):
        """
        Calculate hybrid similarity between job and CV embeddings using RRF.
        
        Args:
            job_embeddings: Dictionary with dense and sparse embeddings for a job
            cv_embeddings: Dictionary with dense and sparse embeddings for a CV
            
        Returns:
            Dictionary with combined score and individual scores
        """
        # Calculate dense similarity
        dense_score = util.cos_sim(job_embeddings['dense'], cv_embeddings['dense']).item()
        
        # Calculate sparse similarity (dot product of matching terms)
        job_sparse = job_embeddings['sparse'].as_object()
        cv_sparse = cv_embeddings['sparse'].as_object()
        
        job_indices = job_sparse["indices"]
        job_values = job_sparse["values"]
        cv_indices = cv_sparse["indices"]
        cv_values = cv_sparse["values"]
        
        # Create dictionaries for faster lookup
        job_dict = dict(zip(job_indices, job_values))
        cv_dict = dict(zip(cv_indices, cv_values))
        
        # Find common indices and calculate dot product
        common_indices = set(job_indices).intersection(set(cv_indices))
        sparse_score = sum(job_dict[idx] * cv_dict[idx] for idx in common_indices)
        
        # Convert scores to ranks for RRF
        dense_rank = 1 + (1 - dense_score)
        sparse_rank = 1 + (1 / (1 + sparse_score))
        
        # Apply RRF formula
        k = self.k_constant
        rrf_score = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))
        
        return {
            'hybrid': rrf_score,
            'dense': dense_score,
            'sparse': sparse_score,
            'dense_rank': dense_rank,
            'sparse_rank': sparse_rank
        }
        
    def batch_rank_cvs(self, job_description, cv_texts):
        """
        Rank multiple CVs against a job description in one efficient operation.
        
        Args:
            job_description: Job description text
            cv_texts: List of CV texts
            
        Returns:
            List of dictionaries with scores for each CV
        """

        # Encode job once
        job_embeddings = self.encode_job(job_description)
        
        # Batch encode all CVs
        cv_embeddings_list = self.batch_encode_cvs(cv_texts)
        
        # Calculate similarities
        results = []
        for cv_embeddings in cv_embeddings_list:
            scores = self.calculate_similarity(job_embeddings, cv_embeddings)
            results.append(scores)
            
        return results
        
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.use_cache:
            self._cache = {}