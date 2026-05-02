from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PyPDF2
import io
import os
from dotenv import load_dotenv

# Load secret key
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI SETUP ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Endpoint 1: Extract Resume
@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return {"resume_text": text}

# Endpoint 2: The Interview Chat (Now with Memory!)
@app.post("/chat")
async def chat(user_message: str = Form(...), resume_text: str = Form(...), role: str = Form(...), chat_history: str = Form(...)):
    prompt = f"""
    You are an expert AI recruiter. The user is applying for: {role}.
    Here is their resume: {resume_text}
    
    Here is the interview history so far:
    {chat_history}
    
    The user just answered: "{user_message}"
    
    Rule 1: Act exactly like a human interviewer.
    Rule 2: Keep your response short (2-3 sentences max).
    Rule 3: Evaluate their answer internally, and then ask the NEXT relevant interview question.
    """
    response = model.generate_content(prompt)
    return {"ai_response": response.text}

# Endpoint 3: The Final Scorecard (The Wow Factor!)
@app.post("/scorecard")
async def generate_scorecard(chat_history: str = Form(...), role: str = Form(...)):
    prompt = f"""
    You are a Senior Tech Recruiter. Review this entire interview transcript for a {role} position.
    Transcript: {chat_history}
    
    Provide a professional evaluation scorecard. Format it cleanly using Markdown.
    Include these exact headings:
    ### 🎯 Final Decision (Hire, No Hire, or Shortlist)
    ### 💪 Top Strengths
    ### 📈 Areas for Improvement
    ### 💡 Expert Advice for Next Time
    """
    response = model.generate_content(prompt)
    return {"scorecard": response.text}