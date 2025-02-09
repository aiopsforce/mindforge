
from typing import Dict, Any, List
from azure.openai import AzureOpenAI
from ...core.base_model import BaseChatModel


class AzureChatModel(BaseChatModel):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
    ):
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}"},
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name, messages=messages
        )
        return response.choices[0].message.content

    def extract_concepts(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "Extract key concepts from the text."},
            {"role": "user", "content": text},
        ]

        response = self.client.chat.completions.create(
            model=self.deployment_name, messages=messages
        )
        #  Handle potential extra spaces and empty strings
        concepts = response.choices[0].message.content.split(",")
        return [c.strip() for c in concepts if c.strip()]
