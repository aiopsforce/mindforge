from openai import OpenAI
from typing import Dict, Any, List
from ...core.base_model import BaseChatModel
from ...utils.errors import ModelError


class OpenAIChatModel(BaseChatModel):
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ModelError(f"OpenAI API error: {e}")

    def extract_concepts(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "Extract key concepts from the text as a comma-separated list."},
            {"role": "user", "content": text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            # Handle potential extra spaces and empty strings
            content = response.choices[0].message.content
            if content:
                concepts = content.split(",")
                return [c.strip() for c in concepts if c.strip()]
            return []
        except Exception as e:
            raise ModelError(f"OpenAI API error: {e}")