import litellm
from typing import Dict, Any, List
from ...core.base_model import BaseChatModel
from ...utils.errors import ModelError


class LiteLLMChatModel(BaseChatModel):
    """Chat model wrapper using litellm for multi-provider support."""

    def __init__(self, model_name: str, api_key: str = None, base_url: str | None = None, extra_params: Dict[str, Any] | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = extra_params or {}

    def _call_completion(self, messages: List[Dict[str, str]]):
        return litellm.completion(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **self.extra_params,
        )

    def _get_content(self, resp) -> str:
        message = resp.choices[0].message
        return message.get("content") if isinstance(message, dict) else message.content

    def generate_response(self, context: Dict[str, Any], query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with memory."},
            {"role": "user", "content": f"Context: {context}\nQuery: {query}"},
        ]
        try:
            response = self._call_completion(messages)
            return self._get_content(response)
        except Exception as e:
            raise ModelError(f"LiteLLM error: {e}")

    def extract_concepts(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "Extract key concepts from the text, separated by commas."},
            {"role": "user", "content": text},
        ]
        try:
            response = self._call_completion(messages)
            content = self._get_content(response)
            return [c.strip() for c in content.split(',') if c.strip()]
        except Exception as e:
            raise ModelError(f"LiteLLM concept extraction error: {e}")
