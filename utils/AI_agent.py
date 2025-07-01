from langchain.chat_models import ChatOllama
from crewai import Agent, Task, Crew

llm = ChatOllama(model="ollama_chat/llama3.2:3b", temperature=0)

def extract_and_check(cv_text):

    extractor_agent = Agent(
        role="Job-Relevant Skill Extractor",
        goal="Extract all skills and years of experience from the given CV TEXT.",
        backstory=(
            "You are an expert in NLP ,  resume parsing and job matching. "
            "You carefully analyze CVs and job descriptions to identify the most relevant skills that can be usefull to any job and the candidate's years of experience, "
            "ensuring that only skills applicable to the job requirements are included."
            ),
        llm = llm,
        verbose=False
        )


    checker_agent = Agent(
        role="Skill and Experience Validator",
        goal="Review the extracted skills and years of experience to ensure they match the actual CV content and are directly related to any job description and not generic or unrelated.",
        backstory=(
            "You are a specialist in candidate-job fit assessment. "
            "You validate that all extracted skills and experience are relevant to the job description, "
            "removing any unrelated or generic information and confirming the years of experience are accurately represented."
            ),
        llm = llm,
        verbose=False
        )
    
    extract_task = Task(
        description=(
        "Extract EVERY skill, tool, framework, and technology explicitly mentioned in the following CV TEXT, "
        "including those mentioned in project descriptions, work experience, and technical skills sections. "
        "ONLY extract years of experience if it is explicitly stated in the text (such as 'X years of experience', 'since 2019', or similar). "
        "Do NOT infer years of experience from education dates, project dates, or any other indirect information. "
        "If years of experience is not explicitly mentioned, always use '0 years of experience'. "
        "List all skills, separated by spaces, after the years of experience. "
        "Format: '<years> years of experience skill1 skill2 skill3 ...'.\n"
        "Example: '2 years of experience python django restapi mysql'.\n"
        "**Do not include any explanation, thoughts, or reasoning. Output only the final answer string.**\n"
        f"CV:\n{cv_text}"
    ),
    expected_output="A single string with experience and relevant skills, e.g., '1 year of experience php laravel mysql restapi docker'.",
    agent=extractor_agent
    )

    check_task = Task(
        description=(
            "Step 2: Review the extracted string below from Step 1. Then, validate it against the full CV.\n\n"
            "Extracted Result:\n<<TASK_RESULT>>\n\n"
            "Full CV:\n"
            f"{cv_text}\n\n"
            "Ensure:\n"
            "- All skills are clearly present in the CV\n"
            "- No hallucinated, unrelated, or AI-related skills are included unless explicitly mentioned\n"
            "- The years of experience are realistically inferred from the text\n\n"
            "- No skills are missing or hallucinated\n"
            "- Do not include any thoughts, reasoning, or explanation. Output only the final answer string.\n"
            "Return the final string in the exact format:\n"
            "'<years> years of experience skill1 skill2 skill3 ...'"
        ),
        expected_output="A single string, e.g., '10 year of experience php laravel mysql restapi docker'.",
        agent=checker_agent,
        depends_on=extract_task
    )

    crew = Crew(
        agents=[extractor_agent, checker_agent],
        tasks=[extract_task, check_task],
        verbose=True
    )
    result = crew.kickoff()
    return str(result)