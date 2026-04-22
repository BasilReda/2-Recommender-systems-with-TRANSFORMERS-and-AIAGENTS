import pandas as pd
import random
from sentence_transformers import InputExample
import numpy as np

def create_contrastive_pairs(job_data=None, n_pairs=10000, return_df=True):
    
    # If no job data provided, try to load from default location
    if job_data is None:
        job_data = pd.read_csv("recommendations.csv")
        print(f"Loaded {len(job_data)} job records")
    
    examples = []
    
    # Define similarity scores
    HIGH_SIM = 0.9  # Components from same job
    LOW_SIM = 0.2   # Components from different jobs
    
    print("Creating positive pairs (components from same job)...")
    # 1. Create positive pairs from the same job listing
    for _, row in job_data.iterrows():
        if pd.isna(row['jobtitle']) or pd.isna(row['jobdescription']) or pd.isna(row['skills']):
            continue
        
        # Job title + description (positive pair)
        examples.append(InputExample(
            texts=[row['jobtitle'], row['jobdescription']],
            label=HIGH_SIM
        ))
        
        # Job description + skills (positive pair)
        examples.append(InputExample(
            texts=[row['jobdescription'], row['skills']],
            label=HIGH_SIM
        ))
        
        # Job title + skills (positive pair)
        examples.append(InputExample(
            texts=[row['jobtitle'], row['skills']],
            label=HIGH_SIM
        ))
    
    print("Creating negative pairs (components from different jobs)...")
    # 2. Create negative pairs by mixing components from different jobs
    # Calculate how many negative pairs we need
    n_negative = min(len(job_data) * 3, n_pairs // 2)
    
    for _ in range(n_negative):
        # Select two different random jobs
        i, j = random.sample(range(len(job_data)), 2)
        
        # Skip if any required data is missing
        if (pd.isna(job_data.iloc[i]['jobtitle']) or 
            pd.isna(job_data.iloc[i]['jobdescription']) or
            pd.isna(job_data.iloc[j]['skills'])):
            continue
        
        # Create negative example: title from one job, skills from another
        examples.append(InputExample(
            texts=[job_data.iloc[i]['jobtitle'], job_data.iloc[j]['skills']],
            label=LOW_SIM
        ))
        
        # Create negative example: description from one job, skills from another
        examples.append(InputExample(
            texts=[job_data.iloc[i]['jobdescription'], job_data.iloc[j]['skills']],
            label=LOW_SIM
        ))
    
    print(f"\nCreated {len(examples)} total examples")
    
    # If DataFrame output is requested, convert examples to DataFrame
    if return_df:
        # Extract data from examples
        data = {
            'text1': [],
            'text2': [],
            'similarity': []
        }
        
        for ex in examples:
            data['text1'].append(ex.texts[0])
            data['text2'].append(ex.texts[1])
            data['similarity'].append(ex.label)
        
        # Create and return DataFrame
        df = pd.DataFrame(data)
        print(f"Returning DataFrame with {len(df)} rows")
        return df
    else:
        # Return the original list of InputExample objects
        return examples


# Function to get train/test split
def get_train_val_data(examples, test_size=0.1, batch_size=16):
    
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    
    # Split into train/validation sets
    train_examples, val_examples = train_test_split(examples, test_size=test_size, random_state=42)
    
    # Create data loaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=batch_size)
    
    print(f"- Training examples: {len(train_examples)}")
    print(f"- Validation examples: {len(val_examples)}")
    
    return train_dataloader, val_dataloader
