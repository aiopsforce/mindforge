import numpy as np
from azure.openai import AzureOpenAI
from ...core.base_model import BaseEmbeddingModel


class AzureEmbeddingModel(BaseEmbeddingModel):
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
        self._dimension = 1536  # Azure OpenAI embedding dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.deployment_name, input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            raise ModelError(f"Azure OpenAI embedding error: {e}")