import os
import google.generativeai as genai
from fastapi import HTTPException, status

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

async def generate_embedding(text: str) -> list[float]:
    """Generates embeddings using Gemini."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini embedding service unavailable: {str(e)}"
        )

async def generate_translation(source_lang: str, target_lang: str, text: str, context: str) -> str:
    """Generates a translation using Gemini 1.5 Flash, incorporating RAG context."""
    system_prompt = (
        f"You are a professional translator. Translate the given text from {source_lang} to {target_lang}. "
        "Return ONLY the translated text without any explanation, greetings, or markdown blocks unless it is part of the translation itself. "
        "Use the following additional context if relevant to improve accuracy or cultural tone:\n"
        f"CONTEXT:\n{context}\n"
    )
    
    try:
        model = genai.GenerativeModel('gemini-3.8-flash', system_instruction=system_prompt)
        response = model.generate_content(text)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini translation service unavailable: {str(e)}"
        )
