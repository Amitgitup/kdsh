import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


class GeminiLLM:
    """
    Thin wrapper around Gemini models (new SDK).
    """

    def __init__(
        self,
        model_name: str = "models/gemini-flash-latest",   # models/gemini-pro-latest
        temperature: float = 0.0,
        max_output_tokens: int = 1536,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.generation_config,
        )

        # Defensive handling
        if response is None:
            return ""

        text = getattr(response, "text", None)
        if not text:
            return ""

        return text.strip()
