from fastapi import FastAPI
from pydantic import BaseModel
from utils.rec_jobs import recommend_jobs
from typing import List
from utils.rec_candidates_copy import recommend_candidates_from_job_description1

app = FastAPI()

class CV(BaseModel):
    user_id: int
    text: str

class SkillsRequest(BaseModel):
    skills : str
    top_k: int = 5

class JobRequest(BaseModel):
    job_description: str
    cvs: List[CV]


@app.post("/recommend_jobs")
def api_recommend_jobs(request: SkillsRequest):
    return recommend_jobs(request.skills, request.top_k)


@app.post("/recommend_candidates")
def api_recommend_candidates(request: JobRequest):
    return recommend_candidates_from_job_description1(job_description=request.job_description,
                                                      top_k=len(request.cvs),
                                                      cvs=request.cvs)