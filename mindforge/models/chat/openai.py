import openai
from typing import Dict, Any, List
from ...core.base_model import BaseChatModel
from ...utils.errors import ModelError

class OpenAIChatModel(BaseChatModel):
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = api_key

    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            raise ModelError(f"OpenAI API error: {e}")


    def extract_concepts(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "Extract key concepts from the text."},
            {"role": "user", "content": text}
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
            # Handle potential extra spaces and empty strings
            concepts = response.choices[0].message.content.split(",")
            return [c.strip() for c in concepts if c.strip()]
        except openai.error.OpenAIError as e:
            raise ModelError(f"OpenAI API error: {e}")