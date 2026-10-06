from typing import List
import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# ============================================================
# 2. NLP SIMILARITY
# ============================================================

def calculate_similarity(resume_text, job_description):
    """
    Calculate similarity between resume and job description
    using TF-IDF and cosine similarity.
    """

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# ============================================================
# 3. SKILL / KEYWORD MATCHING
# ============================================================

def extract_and_match_skills(resume_text, job_description):
    """
    Find important technical skills in the job description
    and check whether they appear in the resume.
    """

    skills_database = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "React",
        "ReactJS",
        "Angular",
        "Angular.js",
        "Node.js",
        "HTML",
        "HTML5",
        "CSS",
        "CSS3",
        "Sass",
        "Redux",
        "Cypress",
        "Selenium",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "NoSQL",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",
        "CI/CD",
        "Jenkins",
        "REST API",
        "API",
        "Django",
        "Flask",
        "FastAPI",
        "Spring Boot",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "LangChain",
        "Power BI",
        "Tableau",
        "Excel",
        "Splunk",
        "Snowflake",
        "JIRA",
        "Confluence",
        "Postman",
        "Agile",
        "Scrum"
    ]

    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    required_skills = []
    matched_skills = []
    missing_skills = []

    # --------------------------------------------------------
    # Find skills mentioned in the job description
    # --------------------------------------------------------

    for skill in skills_database:

        if skill.lower() in job_lower:

            required_skills.append(skill)

            if skill.lower() in resume_lower:
                matched_skills.append(skill)

            else:
                missing_skills.append(skill)

    # --------------------------------------------------------
    # Calculate keyword match percentage
    # --------------------------------------------------------

    if len(required_skills) > 0:

        keyword_match_score = round(
            (len(matched_skills) / len(required_skills)) * 100,
            2
        )

    else:

        keyword_match_score = 0

    return (
        required_skills,
        matched_skills,
        missing_skills,
        keyword_match_score
    )


# ============================================================
# 4. OUTPUT SCHEMA
# ============================================================

class ResumeAnalysis(BaseModel):

    skills: List[str] = Field(
        description="Skills found in the resume"
    )

    missing_skills: List[str] = Field(
        description="Skills missing compared to the job description"
    )

    strengths: List[str] = Field(
        description="Candidate strengths relevant to the job"
    )

    weaknesses: List[str] = Field(
        description="Candidate weaknesses relevant to the job"
    )

    resume_score: int = Field(
        description="ATS score between 0 and 100"
    )

    suggestions: List[str] = Field(
        description="Suggestions to improve the resume"
    )


# ============================================================
# 5. PROMPT
# ============================================================

RESUME_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=[
        "resume_text",
        "job_description"
    ],

    template="""
You are an ATS (Applicant Tracking System).

Analyze the resume STRICTLY against the job description.

Do not assume that the candidate has a skill unless it is
clearly mentioned in the resume.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "skills": [],
    "missing_skills": [],
    "strengths": [],
    "weaknesses": [],
    "resume_score": 0,
    "suggestions": []
}}

Rules:

- skills must be a list of strings.
- missing_skills must be a list of strings.
- strengths must be a list of strings.
- weaknesses must be a list of strings.
- suggestions must be a list of strings.
- resume_score must be an integer between 0 and 100.
- Do not use Markdown.
- Do not use ```json.
- Do not add explanations outside the JSON.

Resume:
{resume_text}

Job Description:
{job_description}
"""
)


# ============================================================
# 6. LOAD RESUME PDF
# ============================================================

def load_pdf(path):
    """
    Extract text from the resume PDF.
    """

    loader = PyPDFLoader(path)

    documents = loader.load()

    resume_text = "\n".join(
        document.page_content
        for document in documents
    )

    return resume_text


# ============================================================
# 7. LOAD JOB DESCRIPTION
# ============================================================

def load_job_description(path):
    """
    Read the job description from a text file.
    """

    loader = TextLoader(path)

    documents = loader.load()

    return documents[0].page_content


# ============================================================
# 8. CLEAN JSON RESPONSE
# ============================================================

def clean_json_response(response):
    """
    Remove Markdown code fences if the model adds them.
    """

    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]

    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


# ============================================================
# 9. ANALYZE RESUME
# ============================================================

def analyze_resume(resume_path, job_description_path):

    # --------------------------------------------------------
    # Load resume
    # --------------------------------------------------------

    resume_text = load_pdf(resume_path)

    # --------------------------------------------------------
    # Load job description
    # --------------------------------------------------------

    job_description = load_job_description(
        job_description_path
    )

    # --------------------------------------------------------
    # Calculate NLP similarity
    # --------------------------------------------------------

    similarity_score = calculate_similarity(
        resume_text,
        job_description
    )

    # --------------------------------------------------------
    # Calculate skill / keyword matching
    # --------------------------------------------------------

    (
        required_skills,
        matched_skills,
        keyword_missing_skills,
        keyword_match_score
    ) = extract_and_match_skills(
        resume_text,
        job_description
    )

    # --------------------------------------------------------
    # Create Groq LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = RESUME_ANALYSIS_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description
    )

    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------------------------
    # Extract response text
    # --------------------------------------------------------

    response_text = response.content

    # --------------------------------------------------------
    # Clean JSON response
    # --------------------------------------------------------

    response_text = clean_json_response(
        response_text
    )

    # --------------------------------------------------------
    # Convert JSON to Python dictionary
    # --------------------------------------------------------

    try:

        data = json.loads(response_text)

    except json.JSONDecodeError:

        print("\nERROR: Groq did not return valid JSON.")

        print("\nModel response:")
        print(response_text)

        return

    # --------------------------------------------------------
    # Validate using Pydantic
    # --------------------------------------------------------

    try:

        result = ResumeAnalysis(**data)

    except Exception as error:

        print("\nERROR: Invalid ResumeAnalysis format.")
        print(error)

        return

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")

    print("=" * 60)
    print("              AI RESUME ANALYZER")
    print("=" * 60)

    # --------------------------------------------------------
    # Score Analysis
    # --------------------------------------------------------

    print("\nScore Analysis")
    print("-" * 60)

    print(f"NLP Similarity : {similarity_score}%")
    print(f"Keyword Match  : {keyword_match_score}%")
    print(f"LLM ATS Score  : {result.resume_score}/100")

    # --------------------------------------------------------
    # Required Skills
    # --------------------------------------------------------

    print("\nRequired Job Skills")
    print("-" * 60)

    if required_skills:

        for skill in required_skills:
            print(f"- {skill}")

    else:

        print("No skills detected from the job description.")

    # --------------------------------------------------------
    # Matched Skills
    # --------------------------------------------------------

    print("\nMatched Job Skills")
    print("-" * 60)

    if matched_skills:

        for skill in matched_skills:
            print(f"✓ {skill}")

    else:

        print("No matching skills found.")

    # --------------------------------------------------------
    # Missing Job Skills
    # --------------------------------------------------------

    print("\nMissing Job Skills")
    print("-" * 60)

    if keyword_missing_skills:

        for skill in keyword_missing_skills:
            print(f"✗ {skill}")

    else:

        print("No missing skills detected.")

    # --------------------------------------------------------
    # AI Skills Found
    # --------------------------------------------------------

    print("\nSkills Found by AI")
    print("-" * 60)

    for skill in result.skills:
        print(f"- {skill}")

    # --------------------------------------------------------
    # AI Missing Skills
    # --------------------------------------------------------

    print("\nMissing Skills Identified by AI")
    print("-" * 60)

    for skill in result.missing_skills:
        print(f"- {skill}")

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    print("\nStrengths")
    print("-" * 60)

    for strength in result.strengths:
        print(f"- {strength}")

    # --------------------------------------------------------
    # Weaknesses
    # --------------------------------------------------------

    print("\nWeaknesses")
    print("-" * 60)

    for weakness in result.weaknesses:
        print(f"- {weakness}")

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    print("\nSuggestions")
    print("-" * 60)

    for suggestion in result.suggestions:
        print(f"- {suggestion}")

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Analysis completed successfully.")
    print("=" * 60)


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    analyze_resume(
        "resume.pdf",
        "job_description.txt"
    )