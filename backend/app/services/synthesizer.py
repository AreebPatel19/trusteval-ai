import os
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, Evolution

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from the environment variables.")

class GeminiSynthesizerLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.3,
                ),
            )
            return schema.model_validate_json(response.text)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.3,
                ),
            )
            return schema.model_validate_json(response.text)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text

    def get_model_name(self) -> str:
        return self.model_name

def generate_layered_questions(contexts: list[list[str]], count_per_context: int = 7) -> list[dict]:
    if not contexts:
        raise ValueError("Contexts list cannot be empty when synthesizing questions.")

    gemini_llm = GeminiSynthesizerLLM(model_name="gemini-2.5-flash")

    evolution_config = EvolutionConfig(
        num_evolutions=2,
        evolutions={
            Evolution.REASONING: 0.30,
            Evolution.MULTICONTEXT: 0.20,
            Evolution.CONSTRAINED: 0.25,
            Evolution.HYPOTHETICAL: 0.25,
        },
    )

    synthesizer = Synthesizer(
        model=gemini_llm,
        evolution_config=evolution_config,
    )

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=count_per_context,
    )

    formatted_goldens = []
    for idx, golden in enumerate(goldens, start=1):
        formatted_goldens.append({
            "id": f"SYNTH-{idx:03d}",
            "category": "Complexity_Layering",
            "input": golden.input.strip(),
            "expected_output": (golden.expected_output or "").strip(),
            "context": golden.context or [],
        })

    return formatted_goldens