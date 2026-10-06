# AI Resume Analyzer

An AI-powered Resume Analyzer that evaluates a resume against a given job description using NLP and Large Language Models (LLMs).

## 🚀 Features

* Extracts text from PDF resumes
* Compares resume content with a job description
* Calculates **TF-IDF similarity score**
* Generates an **AI-based ATS score**
* Identifies relevant skills
* Finds missing skills
* Highlights strengths and weaknesses
* Provides personalized improvement suggestions

## 🛠️ Technologies Used

* Python
* NLP
* TF-IDF
* Large Language Models (LLMs)
* Groq API
* LangChain
* PyPDF
* scikit-learn
* dotenv

## 📂 Project Structure

```text
Resume-Analyzer/
│
├── .env
├── .gitignore
├── app.py
├── job_description.txt
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kaverimyakala/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

**Never upload your actual API key to GitHub.**

### 5. Add your job description

Update:

```text
job_description.txt
```

with the job description you want to analyze your resume against.

### 6. Run the application

```bash
python app.py
```

## 📊 Analysis

The application provides:

* **TF-IDF Similarity Score** – measures the textual similarity between the resume and job description.
* **AI ATS Score** – uses an LLM to evaluate the resume based on the job requirements.
* **Skills Analysis** – identifies matching and missing skills.
* **Resume Improvement Suggestions** – provides recommendations to improve the resume for the target role.

## 🔐 Security

The `.env` file contains the API key and should not be committed to GitHub.

Make sure `.gitignore` contains:

```text
.env
venv/
__pycache__/
```

## 🎯 Future Enhancements

* Batch resume analysis
* Resume ranking
* Excel report generation
* Interactive charts
* Streamlit web interface
* Multiple job-description comparison

## 👩‍💻 Author

**Kaveri Myakala**

AI & Data Science | Python | NLP | Generative AI
