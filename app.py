from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PyPDF2
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return {"resume_text": text}

@app.post("/chat")
async def chat(
    user_message: str = Form(...), 
    resume_text: str = Form(...), 
    role: str = Form(...), 
    job_description: str = Form(...), 
    chat_history: str = Form(...),
    question_count: int = Form(...),
    max_questions: int = Form(...)
):
    # Logic to stop the interview when duration is reached
    if question_count >= max_questions:
         prompt = f"""
         The interview has reached its time limit. 
         Politely thank the candidate for their time, tell them the interview is now complete, and ask them to click the 'Finish & Get Scorecard' button. Do not ask any more questions.
         """
    else:
        prompt = f"""
        You are an expert AI recruiter for the {role} position.
        
        Candidate's Resume: {resume_text}
        Job Description Requirements: {job_description}
        Interview History: {chat_history}
        Candidate just said: "{user_message}"
        
        This is question {question_count + 1} out of {max_questions}.
        
        Rules:
        1. Act exactly like a tough but fair human interviewer.
        2. Keep your response short (2-3 sentences max).
        3. Ask the NEXT relevant interview question. The question MUST directly test if their resume skills match the Job Description. Ask technical or situational questions related to the field.
        """
        
    response = model.generate_content(prompt)
    return {"ai_response": response.text}

@app.post("/scorecard")
async def generate_scorecard(chat_history: str = Form(...), role: str = Form(...), job_description: str = Form(...)):
    prompt = f"""
    You are a Senior Tech Recruiter. Review this interview transcript for a {role} position.
    Job Description: {job_description}
    Transcript: {chat_history}
    
    Format a highly detailed evaluation scorecard using Markdown. Include:
    ### 🎯 Final Decision (Hire, Shortlist, or Reject)
    ### 💼 JD vs Resume Alignment (How well did they prove they fit the specific job description?)
    ### 💪 Top Strengths
    ### 📉 Areas for Improvement & Verbal Confidence
    ### 💡 Next Steps / Advice
    """
    response = model.generate_content(prompt)
    return {"scorecard": response.text}