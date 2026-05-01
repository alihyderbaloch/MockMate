from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PyPDF2
import io

app = FastAPI()

# frontend to talk to this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI SETUP ---

GEMINI_API_KEY = "" 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- ENDPOINT 1: Read the PDF Resume ---
@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    # This reads the uploaded PDF and extracts the text
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return {"resume_text": text}

# --- ENDPOINT 2: Chat with the AI ---
@app.post("/chat")
async def chat(user_message: str = Form(...), resume_text: str = Form(...), role: str = Form(...)):
    # "System Prompt" that gives the AI its personality and rules
    prompt = f"""
    You are an expert AI recruiter. The user is applying for the role of {role}.
    Here is their resume data: {resume_text}
    
    The user just said: "{user_message}"
    
    Rule 1: Act exactly like a human interviewer.
    Rule 2: Keep your response short (2-3 sentences maximum).
    Rule 3: Ask ONE relevant interview question based on their resume or role, or respond to their answer and ask the next question.
    """
    
    # Send the prompt to Gemini and get the response
    response = model.generate_content(prompt)
    return {"ai_response": response.text}


# uvicorn backend:app --reload