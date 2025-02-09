
import numpy as np
import openai
from ...core.base_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = api_key
        self._dimension = 1536  #  text-embedding-3-small dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = openai.Embedding.create(model=self.model_name, input=text)
            return np.array(response.data[0].embedding)
        except openai.error.OpenAIError as e:
            raise ModelError(f"OpenAI API error: {e}")