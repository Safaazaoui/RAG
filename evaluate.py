import os
import time
from typing import List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env
client = OpenAI()  # picks up OPENAI_API_KEY automatically
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class EvaluationResult(BaseModel):
    faithfulness_score: float = Field(..., description="Score between 0.0 and 1.0 indicating if context supports answer.")
    is_hallucinated: bool = Field(..., description="True if answer introduces claims not in context.")
    reasoning: str = Field(..., description="Brief explanation for the score.")

EVALUATION_PROMPT = """
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Analyze the provided Context and the Generated Answer to evaluate Faithfulness.

Context:
{context}

Generated Answer:
{answer}

Determine if the Generated Answer is strictly supported by the Context. 
Output a score from 0.0 (completely ungrounded/hallucinated) to 1.0 (fully faithful).
"""

def evaluate_faithfulness(context: str, answer: str) -> EvaluationResult:
    """
    Uses OpenAI Structured Outputs to grade RAG answer faithfulness.
    """
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You evaluate AI outputs for strict faithfulness to context."},
            {"role": "user", "content": EVALUATION_PROMPT.format(context=context, answer=answer)}
        ],
        response_format=EvaluationResult,
    )
    return response.choices[0].message.parsed

if __name__ == "__main__":
    # Test Evaluation Benchmark
    test_cases: List[Dict[str, str]] = [
        {
            "query": "What database does BodIQ AI use?",
            "context": "BodIQ AI uses PostgreSQL hosted on Supabase with Row-Level Security policies.",
            "answer": "BodIQ AI runs on a PostgreSQL database provided by Supabase."
        },
        {
            "query": "What model handles barcode scanning?",
            "context": "BodIQ AI uses OpenFoodFacts API for barcode scanning and Gemini Vision for photos.",
            "answer": "OpenAI GPT-4 handles barcode scanning."
        }
    ]

    print("🔍 Running RAG Evaluation Suite...\n" + "="*45)
    for idx, test in enumerate(test_cases, 1):
        res = evaluate_faithfulness(test["context"], test["answer"])
        print(f"Test #{idx}: {test['query']}")
        print(f"  • Faithfulness Score : {res.faithfulness_score}")
        print(f"  • Hallucinated       : {res.is_hallucinated}")
        print(f"  • Evaluator Rationale: {res.reasoning}\n")