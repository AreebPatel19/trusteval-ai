import os
from pypdf import PdfReader
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from the environment variables.")

client = genai.Client(api_key=api_key)

class ExtractedContexts(BaseModel):
    rules: list[str]

def extract_raw_text(pdf_path: str | None = None, raw_prompt: str | None = None) -> str:
    text_segments = []
    
    if pdf_path and os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_segments.append(page_text.strip())
                
    if raw_prompt and raw_prompt.strip():
        text_segments.append(f"System Prompt / Rules:\n{raw_prompt.strip()}")
        
    combined_text = "\n\n".join(text_segments).strip()
    if not combined_text:
        raise ValueError("No readable text could be extracted from the provided inputs.")
        
    return combined_text

def extract_entities_and_rules(raw_text: str) -> list[list[str]]:
    prompt = (
        "Act as a Senior QA Architect. Read the following system documentation and prompt, "
        "then extract the core contracts, business rules, entity relationships, and strict "
        "operational boundaries. Make each extracted item a self-contained rule or context "
        "block. Generate at least 10 distinct context blocks if the source material allows.\n\n"
        f"Documentation:\n{raw_text}"
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedContexts,
            temperature=0.1,
        ),
    )
    
    parsed_data = ExtractedContexts.model_validate_json(response.text)
    return [[rule] for rule in parsed_data.rules if rule.strip()]