# Resume Recommendation API

This project provides a FastAPI-based backend for recommending jobs to users and ranking candidates for jobs using semantic similarity of resumes and job descriptions.

## Features
- Recommend jobs based on user skills
- Recommend candidates based on job description and a list of CVs
- Uses Sentence Transformers for semantic similarity
- REST API with FastAPI

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

## API Endpoints

### `/recommend_jobs`
- **POST**: Recommend jobs based on user skills
- **Request Body Example:**
```json
{
  "skills": "python machine learning",
  "top_k": 5
}
```

### `/recommend_candidates`
- **POST**: Recommend candidates based on job description and a list of CVs
- **Request Body Example:**
```json
{
  "job_description": "backend developer",
  "cvs": [
    {"user_id": 1, "text": "CV text 1..."},
    {"user_id": 2, "text": "CV text 2..."}
  ]
}
```

## Project Structure
- `app.py` : Main FastAPI app and API endpoints
- `utils/rec_jobs.py` : Job recommendation logic
- `utils/rec_candidates_copy.py` : Candidate recommendation logic
- `src/` : Helper modules for data processing and embedding

## License
MIT
