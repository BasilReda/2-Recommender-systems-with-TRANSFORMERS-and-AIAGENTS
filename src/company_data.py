import PyPDF2
import os
import pandas as pd
import re
import spacy
import pickle
from sentence_transformers import SentenceTransformer

def extract_text_description(path =  "cv"):
    extracted_data = []

    pdf_files = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]

    for file in pdf_files:
        file_path = os.path.join(path , file)

        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            
            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

            text = ' '.join(text.split())

            extracted_data.append({
                'filename': file , 'text': text
            })

    df = pd.DataFrame(extracted_data)   
    return df 

def clean_resume_text(text):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove contact information
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)  # Email
    text = re.sub(r'(\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}', ' ', text)  # Phone
    
    # 3. Remove URLs and social media handles
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'linkedin\.com\S*|github\.com\S*|twitter\.com\S*', ' ', text)
    
    # 4. Remove locations and addresses
    location_patterns = [
        r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, State
        r'\b[A-Z][a-z]+\s+[A-Z]{2}\s+\d{5}\b',  # City State ZIP
    ]
    for pattern in location_patterns:
        text = re.sub(pattern, ' ', text)
    
    # 5. Remove dates (but preserve skill versions like 'Python 3')
    text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' ', text)
    text = re.sub(r'\b\d{4}-\d{2,4}\b', ' ', text)
    
    # 6. Remove special characters but keep important ones for skills
    text = text.replace('|', ' ').replace('•', ' ').replace('–', ' ').replace('/', ' ')
    
    # 7. Process with spaCy for advanced cleaning
    doc = nlp(text)
    
    # 8. Keep only relevant tokens 
    tokens = []
    for token in doc:
        # Skip if token is punctuation, stopword, or just whitespace
        if (
            not token.is_punct
            and not token.is_stop
            and not token.is_space
            and token.text.strip()
            and len(token.text.strip()) > 1 or token.text.strip().isdigit()
        ):
            # Lemmatize the token
            tokens.append(token.text)
    
    # 9. Join tokens back into cleaned text
    cleaned_text = " ".join(tokens)
    
    # 10. Remove extra whitespace
    cleaned_text = " ".join(cleaned_text.split())
    # return the cleaned_text in the shape of pandas dataframe with each has its own index
    return cleaned_text

def clean_resume_dataframe(df):
    cleaned_df = df.copy()
    
    cleaned_df['Resume'] = cleaned_df['text'].apply(clean_resume_text)
    
    print(f"Successfully cleaned {len(cleaned_df)} resume texts")
    
    return cleaned_df

def save_embedded_cv(df , output_file='embedded_evaluated2_cv.pkl' , model = "job_embeddings_model_evaluated2"):
    model = SentenceTransformer(model)

    embedding_dic = {}

    for idx , row in  df.iterrows():
        cv_filename = row['filename']
        text = row['Resume']

        embedding = model.encode([text])[0]

        embedding_dic[cv_filename] = embedding

    with open(output_file , "wb") as f:
        pickle.dump(embedding_dic, f)

    return embedding_dic